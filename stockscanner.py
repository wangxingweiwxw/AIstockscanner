import streamlit as st

# 必须在任何其他Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="AI Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import traceback
import json
from typing import Optional, Dict, Any, List
import asyncio
import re
import sys
import os

# 添加services目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

# 导入services模块
try:
    from services.stock_analyzer_service import StockAnalyzerService
    from services.stock_data_provider import StockDataProvider
    from services.technical_indicator import TechnicalIndicator
    from services.stock_scorer import StockScorer
    from utils.logger import get_logger
    logger = get_logger()
except ImportError as e:
    st.error(f"❌ 导入services模块失败: {e}")
    st.info("请确保services目录下的所有文件都存在")
    logger = None

# Helper Functions
def format_stock_code(code, market_type='A'):
    """
    Formats the stock code based on market type.
    """
    if isinstance(code, str):
        code = code.strip().upper()
        if market_type == 'A':
            # Remove potential prefixes for A-shares
            code = re.sub(r'^(SH|SZ|BJ)', '', code)
            return f"{int(code):06d}"
        elif market_type == 'HK':
            # Hong Kong stocks: add .HK suffix if not present
            if not code.endswith('.HK'):
                return f"{code}.HK"
            return code
        elif market_type == 'US':
            # US stocks: keep as is
            return code
    return code

class EnhancedStockAIAnalyzer:
    """
    Enhanced AI Analyzer that integrates with services
    """
    def __init__(self):
        try:
            # Initialize the main analyzer service
            self.analyzer_service = StockAnalyzerService(
                custom_api_url=st.secrets.get("api_url"),
                custom_api_key=st.secrets.get("api_key"),
                custom_api_model=st.secrets.get("api_model", "glm-4"),
                custom_api_timeout=300
            )
            
            # Initialize individual services for direct access
            self.data_provider = StockDataProvider()
            self.indicator = TechnicalIndicator()
            self.scorer = StockScorer()
            
            # Initialize AI analyzer
            from services.ai_analyzer import AIAnalyzer
            self.ai_analyzer = AIAnalyzer(
                custom_api_url=st.secrets.get("api_url"),
                custom_api_key=st.secrets.get("api_key"),
                custom_api_model=st.secrets.get("api_model", "glm-4"),
                custom_api_timeout=300
            )
            
        except Exception as e:
            st.error(f"❌ AI分析器初始化失败: {e}")
            self.analyzer_service = None
            self.data_provider = None
            self.indicator = None
            self.scorer = None
            self.ai_analyzer = None
    
    async def analyze_single_stock(self, stock_code: str, stock_name: str, market_type: str = 'A') -> dict:
        """
        Analyze a single stock using the services
        """
        if not self.analyzer_service:
            return {"error": "分析器未初始化"}
        
        try:
            # Use the analyzer service for comprehensive analysis
            analysis_results = []
            async for result in self.analyzer_service.analyze_stock(stock_code, market_type, stream=False):
                if isinstance(result, str):
                    try:
                        data = json.loads(result)
                        analysis_results.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Extract the main analysis result
            if analysis_results:
                main_result = analysis_results[0]  # First result is usually the basic analysis
                
                # Add AI analysis if available
                if self.ai_analyzer and self.ai_analyzer.client:
                    try:
                        # Get stock data for AI analysis
                        df = await self.data_provider.get_stock_data(stock_code, market_type)
                        if not df.empty:
                            # Calculate technical indicators for AI analysis
                            df_with_indicators = self.indicator.calculate_indicators(df)
                            
                            # Get AI analysis
                            async for ai_result in self.ai_analyzer.get_ai_analysis(df_with_indicators, stock_code, market_type, stream=False):
                                if isinstance(ai_result, str):
                                    try:
                                        ai_data = json.loads(ai_result)
                                        if "ai_analysis" in ai_data:
                                            main_result["ai_analysis"] = ai_data["ai_analysis"]
                                            break
                                        elif "error" in ai_data:
                                            # If AI analysis fails, use fallback
                                            main_result["ai_analysis"] = self._generate_fallback_analysis(df_with_indicators, stock_code, market_type)
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        main_result["ai_analysis"] = self._generate_fallback_analysis(df_with_indicators, stock_code, market_type) if 'df_with_indicators' in locals() else f"AI分析失败: {str(e)}"
                else:
                    main_result["ai_analysis"] = "AI分析器未初始化"
                
                return main_result
            else:
                return {"error": "无法获取分析结果"}
                
        except Exception as e:
            return {"error": f"分析股票时出错: {str(e)}"}
    
    async def batch_analyze_stocks(self, stock_codes: List[str], market_type: str = 'A', min_score: int = 0) -> List[dict]:
        """
        Batch analyze multiple stocks
        """
        if not self.analyzer_service:
            return [{"error": "分析器未初始化"}]
        
        try:
            results = []
            async for result in self.analyzer_service.scan_stocks(stock_codes, market_type, min_score, stream=False):
                if isinstance(result, str):
                    try:
                        data = json.loads(result)
                        if "stock_code" in data and "error" not in data:
                            results.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return results
            
        except Exception as e:
            return [{"error": f"批量分析时出错: {str(e)}"}]
    
    async def get_ai_analysis_only(self, stock_code: str, market_type: str = 'A') -> str:
        """
        Get only AI analysis for a stock
        """
        if not self.ai_analyzer or not self.ai_analyzer.client:
            return "AI分析器未初始化"
        
        try:
            # Get stock data
            df = await self.data_provider.get_stock_data(stock_code, market_type)
            
            if df.empty:
                return "无法获取股票数据"
            
            # Calculate technical indicators
            df_with_indicators = self.indicator.calculate_indicators(df)
            
            # Get AI analysis with better error handling
            try:
                async for ai_result in self.ai_analyzer.get_ai_analysis(df_with_indicators, stock_code, market_type, stream=False):
                    if isinstance(ai_result, str):
                        try:
                            ai_data = json.loads(ai_result)
                            if "ai_analysis" in ai_data:
                                return ai_data["ai_analysis"]
                            elif "error" in ai_data:
                                # Return a more user-friendly error message
                                error_msg = ai_data["error"]
                                if "部署失败" in error_msg or "deployment failed" in error_msg.lower():
                                    return self._generate_fallback_analysis(df_with_indicators, stock_code, market_type)
                                else:
                                    return f"AI分析失败: {error_msg}"
                        except json.JSONDecodeError:
                            continue
                
                return "AI分析结果为空"
                
            except Exception as ai_error:
                if logger:
                    logger.error(f"AI分析调用失败: {str(ai_error)}")
                # 如果AI分析失败，提供备用分析
                return self._generate_fallback_analysis(df_with_indicators, stock_code, market_type)
            
        except Exception as e:
            return f"获取股票数据失败: {str(e)}"
    
    def _generate_fallback_analysis(self, df: pd.DataFrame, stock_code: str, market_type: str = 'A') -> str:
        """
        生成备用分析（当AI分析失败时）
        """
        try:
            if df.empty:
                return "无法生成分析：数据不足"
            
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # 计算基本指标
            current_price = latest.get('Close', 0)
            price_change = current_price - previous.get('Close', current_price)
            price_change_pct = (price_change / previous.get('Close', current_price)) * 100 if previous.get('Close', current_price) != 0 else 0
            
            # 技术指标分析
            rsi = latest.get('RSI', 50)
            ma5 = latest.get('MA5', current_price)
            ma20 = latest.get('MA20', current_price)
            ma60 = latest.get('MA60', current_price)
            
            # 生成分析
            market_names = {'A': 'A股', 'HK': '港股', 'US': '美股'}
            market_name = market_names.get(market_type, market_type)
            
            analysis = f"""
## 📊 {market_name}股票 {stock_code} 技术分析

### 价格信息
- 当前价格: ¥{current_price:.2f}
- 价格变动: {price_change:+.2f} ({price_change_pct:+.2f}%)

### 技术指标分析
- RSI(14): {rsi:.2f} {'(超买)' if rsi > 70 else '(超卖)' if rsi < 30 else '(正常)'}
- MA5: ¥{ma5:.2f}
- MA20: ¥{ma20:.2f}
- MA60: ¥{ma60:.2f}

### 趋势分析
"""
            
            # 趋势判断
            if ma5 > ma20 > ma60:
                analysis += "- 短期、中期、长期均线呈多头排列，趋势向上\n"
            elif ma5 < ma20 < ma60:
                analysis += "- 短期、中期、长期均线呈空头排列，趋势向下\n"
            else:
                analysis += "- 均线排列混乱，趋势不明确\n"
            
            # RSI分析
            if rsi > 70:
                analysis += "- RSI处于超买区域，需注意回调风险\n"
            elif rsi < 30:
                analysis += "- RSI处于超卖区域，可能存在反弹机会\n"
            else:
                analysis += "- RSI处于正常区域，市场相对平衡\n"
            
            # 投资建议
            analysis += "\n### 投资建议\n"
            if price_change_pct > 5 and rsi > 70:
                analysis += "⚠️ 注意风险：股价涨幅较大且RSI超买，建议谨慎操作\n"
            elif price_change_pct < -5 and rsi < 30:
                analysis += "💡 关注机会：股价跌幅较大且RSI超卖，可关注反弹机会\n"
            elif ma5 > ma20 > ma60:
                analysis += "📈 趋势向好：均线多头排列，可考虑逢低买入\n"
            elif ma5 < ma20 < ma60:
                analysis += "📉 趋势向下：均线空头排列，建议观望或减仓\n"
            else:
                analysis += "⏸️ 震荡整理：趋势不明确，建议观望为主\n"
            
            analysis += "\n*注：此分析基于技术指标，仅供参考，投资需谨慎*"
            
            return analysis
            
        except Exception as e:
            return f"生成备用分析失败: {str(e)}"
    
    def get_technical_analysis(self, stock_code: str, market_type: str = 'A') -> dict:
        """
        Get detailed technical analysis using the services
        """
        try:
            # Get stock data
            df = asyncio.run(self.data_provider.get_stock_data(stock_code, market_type))
            
            if df.empty:
                return {"error": "无法获取股票数据"}
            
            # Calculate technical indicators
            df_with_indicators = self.indicator.calculate_indicators(df)
            
            # Get the latest indicators
            latest = df_with_indicators.iloc[-1]
            
            return {
                "MA5": float(latest.get('MA5', 0)),
                "MA20": float(latest.get('MA20', 0)),
                "MA60": float(latest.get('MA60', 0)),
                "RSI": float(latest.get('RSI', 0)),
                "MACD": float(latest.get('MACD', 0)),
                "Signal": float(latest.get('Signal', 0)),
                "BB_Upper": float(latest.get('BB_Upper', 0)),
                "BB_Lower": float(latest.get('BB_Lower', 0)),
                "Volume_Ratio": float(latest.get('Volume_Ratio', 1)),
                "Volatility": float(latest.get('Volatility', 0)),
                "ATR": float(latest.get('ATR', 0))
            }
            
        except Exception as e:
            return {"error": f"技术分析出错: {str(e)}"}

def ai_stock_scanner():
    st.title("📈 AI Stock Scanner")
    st.markdown("---")
    
    # Initialize the enhanced analyzer
    analyzer = EnhancedStockAIAnalyzer()
    
    if not analyzer.analyzer_service:
        st.error("❌ 分析器初始化失败，请检查配置")
        return
    
    # Show AI analyzer status
    if analyzer.ai_analyzer and analyzer.ai_analyzer.client:
        # Add AI status test button
        if st.button("🔍 测试AI服务状态", help="点击测试AI服务是否正常工作"):
            with st.spinner("正在测试AI服务..."):
                try:
                    # 创建一个简单的测试数据
                    test_data = pd.DataFrame({
                        'Close': [100, 101, 102, 103, 104],
                        'Open': [99, 100, 101, 102, 103],
                        'High': [101, 102, 103, 104, 105],
                        'Low': [98, 99, 100, 101, 102],
                        'Volume': [1000, 1100, 1200, 1300, 1400]
                    })
                    
                    # 计算技术指标
                    test_data_with_indicators = analyzer.indicator.calculate_indicators(test_data)
                    
                    # 测试AI分析 - 使用正确的异步处理方式
                    async def test_ai_service():
                        success = False
                        try:
                            async for result in analyzer.ai_analyzer.get_ai_analysis(test_data_with_indicators, "TEST", "A", stream=False):
                                if isinstance(result, str):
                                    try:
                                        data = json.loads(result)
                                        if "ai_analysis" in data and data["ai_analysis"]:
                                            success = True
                                            break
                                    except:
                                        continue
                        except Exception as e:
                            st.error(f"AI服务调用失败: {str(e)}")
                            return False
                        
                        return success
                    
                    # 运行异步测试
                    test_result = asyncio.run(test_ai_service())
                    
                    if test_result:
                        st.success("✅ AI服务正常，可以进行分析")
                    else:
                        st.warning("⚠️ AI服务响应异常，但备用分析可用")
                        
                except Exception as e:
                    st.error(f"❌ AI服务测试失败: {str(e)}")
                    st.info("💡 将使用备用技术分析功能")
    else:
        st.warning("⚠️ AI分析器未初始化，AI分析功能将不可用")
        st.info("💡 将使用备用技术分析功能")
    
    # Market selection
    st.subheader("🎯 选择市场")
    market_options = {
        "A股": "A",
        "港股": "HK", 
        "美股": "US"
    }
    
    selected_market = st.selectbox(
        "选择市场类型:",
        options=list(market_options.keys()),
        index=0
    )
    
    market_type = market_options[selected_market]
    
    # Stock input section
    st.subheader("📊 输入股票代码")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**方式一：单个股票分析**")
        single_stock_code = st.text_input(
            "输入单个股票代码:",
            placeholder="A股: 000001, 港股: 00700, 美股: AAPL"
        )
    
    with col2:
        st.markdown("**方式二：批量股票分析**")
        batch_stock_codes = st.text_input(
            "输入多个股票代码 (逗号分隔):",
            placeholder="000001,600519,00700,AAPL"
        )
    
    # Combine stock codes
    all_stock_codes = []
    
    if single_stock_code:
        all_stock_codes.append(single_stock_code.strip())
    
    if batch_stock_codes:
        codes = [code.strip() for code in batch_stock_codes.split(',') if code.strip()]
        all_stock_codes.extend(codes)
    
    # Remove duplicates and limit to 10
    all_stock_codes = list(set(all_stock_codes))[:10]
    
    # Display selected stocks
    if all_stock_codes:
        st.subheader("📋 待分析股票")
        for i, code in enumerate(all_stock_codes, 1):
            st.write(f"{i}. {code}")
    
    # Analysis options
    st.subheader("⚙️ 分析选项")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_score = st.slider("最低评分阈值:", 0, 100, 50)
    
    with col2:
        analysis_type = st.selectbox(
            "分析类型:",
            ["单只股票详细分析", "批量扫描分析", "纯AI分析"]
        )
    
    with col3:
        include_technical = st.checkbox("包含技术分析", value=True)
    
    with col4:
        include_ai_analysis = st.checkbox("包含AI分析", value=True)
    
    # Market code format help
    with st.expander("💡 股票代码格式说明"):
        st.markdown("""
        **A股代码格式:**
        - 上海证券交易所: 600xxx, 601xxx, 603xxx, 688xxx
        - 深圳证券交易所: 000xxx, 002xxx, 300xxx
        - 北京证券交易所: 430xxx, 830xxx, 870xxx
        
        **港股代码格式:**
        - 腾讯控股: 00700
        - 阿里巴巴: 09988
        - 美团: 03690
        
        **美股代码格式:**
        - 苹果: AAPL
        - 微软: MSFT
        - 特斯拉: TSLA
        
        **AI分析功能:**
        - 选择"包含AI分析"可获得AI智能分析
        - 选择"纯AI分析"仅获取AI分析结果
        - AI分析基于技术指标和价格数据提供投资建议
        - 如果AI服务不可用，将自动使用备用技术分析
        - 点击"测试AI服务状态"可检查AI服务是否正常
        """)
    
    # Start analysis
    if st.button("🚀 开始分析", type="primary"):
        if not all_stock_codes:
            st.warning("请输入至少一只股票代码")
            return
        
        st.subheader("📈 分析结果")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if analysis_type == "单只股票详细分析":
                # Single stock detailed analysis
                for i, stock_code in enumerate(all_stock_codes):
                    progress = (i + 1) / len(all_stock_codes)
                    progress_bar.progress(progress)
                    status_text.text(f"正在分析 {stock_code}...")
                    
                    # Use stock code as name if no name available
                    stock_name = stock_code
                    
                    # Analyze stock
                    result = asyncio.run(analyzer.analyze_single_stock(stock_code, stock_name, market_type))
                    
                    if "error" in result:
                        st.error(f"{stock_code}: {result['error']}")
                    else:
                        # Display results
                        with st.expander(f"📊 {stock_name} - 评分: {result.get('score', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("当前价格", f"{result.get('price', 'N/A')}")
                                st.metric("涨跌幅", f"{result.get('change_percent', 'N/A'):.2f}%")
                                st.metric("RSI", f"{result.get('rsi', 'N/A'):.2f}")
                                st.metric("推荐", result.get('recommendation', 'N/A'))
                            
                            with col2:
                                st.metric("MA趋势", result.get('ma_trend', 'N/A'))
                                st.metric("MACD信号", result.get('macd_signal', 'N/A'))
                                st.metric("成交量状态", result.get('volume_status', 'N/A'))
                                
                                # Color-coded score
                                score = result.get('score', 0)
                                if score >= 80:
                                    st.success(f"评分: {score}")
                                elif score >= 60:
                                    st.info(f"评分: {score}")
                                else:
                                    st.error(f"评分: {score}")
                            
                            # Technical analysis details
                            if include_technical:
                                st.subheader("🔧 技术指标详情")
                                tech_analysis = analyzer.get_technical_analysis(stock_code, market_type)
                                
                                if "error" not in tech_analysis:
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.write("**移动平均线**")
                                        st.write(f"MA5: {tech_analysis['MA5']:.2f}")
                                        st.write(f"MA20: {tech_analysis['MA20']:.2f}")
                                        st.write(f"MA60: {tech_analysis['MA60']:.2f}")
                                    
                                    with col2:
                                        st.write("**技术指标**")
                                        st.write(f"RSI: {tech_analysis['RSI']:.2f}")
                                        st.write(f"MACD: {tech_analysis['MACD']:.2f}")
                                        st.write(f"Signal: {tech_analysis['Signal']:.2f}")
                                    
                                    with col3:
                                        st.write("**其他指标**")
                                        st.write(f"成交量比率: {tech_analysis['Volume_Ratio']:.2f}")
                                        st.write(f"波动率: {tech_analysis['Volatility']:.2f}%")
                                        st.write(f"ATR: {tech_analysis['ATR']:.2f}")
                                else:
                                    st.error(f"技术分析失败: {tech_analysis['error']}")
                            
                            # AI analysis details
                            if include_ai_analysis:
                                st.subheader("🤖 AI智能分析")
                                
                                # Check if AI analysis is already in the result
                                if "ai_analysis" in result and result["ai_analysis"]:
                                    ai_content = result["ai_analysis"]
                                    if ai_content != "AI分析器未初始化" and not ai_content.startswith("AI分析失败"):
                                        st.markdown(ai_content)
                                    else:
                                        st.warning("AI分析未完成，正在重新获取...")
                                        # Try to get AI analysis separately
                                        ai_analysis = asyncio.run(analyzer.get_ai_analysis_only(stock_code, market_type))
                                        if ai_analysis and ai_analysis != "AI分析器未初始化":
                                            st.markdown(ai_analysis)
                                        else:
                                            st.error("无法获取AI分析结果")
                                else:
                                    st.info("正在获取AI分析...")
                                    # Get AI analysis separately
                                    ai_analysis = asyncio.run(analyzer.get_ai_analysis_only(stock_code, market_type))
                                    if ai_analysis and ai_analysis != "AI分析器未初始化":
                                        st.markdown(ai_analysis)
                                    else:
                                        st.error("无法获取AI分析结果")
            
            elif analysis_type == "纯AI分析":
                # Pure AI analysis only
                st.subheader("🤖 AI智能分析")
                
                for i, stock_code in enumerate(all_stock_codes):
                    progress = (i + 1) / len(all_stock_codes)
                    progress_bar.progress(progress)
                    status_text.text(f"正在获取 {stock_code} 的AI分析...")
                    
                    with st.expander(f"📊 {stock_code} - AI分析"):
                        st.info("正在获取AI分析...")
                        ai_analysis = asyncio.run(analyzer.get_ai_analysis_only(stock_code, market_type))
                        if ai_analysis and ai_analysis != "AI分析器未初始化":
                            st.markdown(ai_analysis)
                        else:
                            st.error("无法获取AI分析结果")
            
            else:
                # Batch analysis
                status_text.text("正在进行批量分析...")
                
                results = asyncio.run(analyzer.batch_analyze_stocks(all_stock_codes, market_type, min_score))
                
                if results:
                    # Create results DataFrame
                    results_data = []
                    for result in results:
                        if "error" not in result:
                            results_data.append({
                                "股票代码": result.get('stock_code', ''),
                                "评分": result.get('score', 0),
                                "推荐": result.get('recommendation', ''),
                                "价格": result.get('price', 0),
                                "RSI": f"{result.get('rsi', 'N/A'):.2f}",
                                "MA趋势": result.get('ma_trend', ''),
                                "MACD信号": result.get('macd_signal', ''),
                                "成交量状态": result.get('volume_status', '')
                            })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        
                        # Sort by score
                        results_df = results_df.sort_values('评分', ascending=False)
                        
                        # Display results
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📊 分析统计")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("分析股票数", len(results_data))
                        
                        with col2:
                            avg_score = results_df['评分'].mean()
                            st.metric("平均评分", f"{avg_score:.1f}")
                        
                        with col3:
                            high_score_count = len(results_df[results_df['评分'] >= 80])
                            st.metric("高分股票", high_score_count)
                        
                        with col4:
                            buy_recommendations = len(results_df[results_df['推荐'].str.contains('推荐', na=False) & ~results_df['推荐'].str.contains('不推荐', na=False)])
                            st.metric("推荐股票", buy_recommendations)
                        
                        # AI Analysis for top stocks
                        if include_ai_analysis:
                            st.subheader("🤖 AI智能分析 - TOP3股票")
                            
                            # Get top 3 stocks for AI analysis
                            top_stocks = results_df.head(3)
                            
                            for idx, row in top_stocks.iterrows():
                                stock_code = row['股票代码']
                                score = row['评分']
                                
                                with st.expander(f"📈 {stock_code} - 评分: {score} - AI分析"):
                                    ai_analysis = asyncio.run(analyzer.get_ai_analysis_only(stock_code, market_type))
                                    if ai_analysis and ai_analysis != "AI分析器未初始化":
                                        st.markdown(ai_analysis)
                                    else:
                                        st.error("无法获取AI分析结果")
                    else:
                        st.warning("没有找到符合条件的股票")
                else:
                    st.error("批量分析失败")
        
        except Exception as e:
            st.error(f"分析过程中出错: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.empty()
            status_text.empty()
            st.success("✅ 分析完成!")

def main():
    # Check for secrets
    if 'api_key' not in st.secrets or 'api_url' not in st.secrets:
        st.error("❌ 缺少API配置。请在 `.streamlit/secrets.toml` 中添加 `api_key` 和 `api_url`")
        st.info("示例配置:\n```toml\napi_key = \"YOUR_API_KEY\"\napi_url = \"YOUR_API_URL\"\napi_model = \"glm-4\"\n```")
        return
    
    # Add some styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    ai_stock_scanner()

if __name__ == "__main__":
    main()