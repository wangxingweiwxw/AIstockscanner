import json
import pandas as pd
from typing import AsyncGenerator, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import get_logger

# 获取日志器
logger = get_logger()

class AIAnalyzer:
    """
    AI分析器服务
    负责使用AI模型对股票数据进行深度分析
    """
    
    def __init__(self, custom_api_url: Optional[str] = None, 
                 custom_api_key: Optional[str] = None,
                 custom_api_model: Optional[str] = None,
                 custom_api_timeout: Optional[int] = 300):
        """
        初始化AI分析器
        
        Args:
            custom_api_url: 自定义API URL
            custom_api_key: 自定义API密钥
            custom_api_model: 自定义API模型
            custom_api_timeout: 自定义API超时时间
        """
        try:
            import streamlit as st
            
            self.api_url = custom_api_url or st.secrets.get("api_url")
            self.api_key = custom_api_key or st.secrets.get("api_key")
            self.api_model = custom_api_model or st.secrets.get("api_model", "glm-4")
            self.api_timeout = custom_api_timeout or 300
            
            if not self.api_key or not self.api_url:
                logger.error("缺少API配置")
                self.client = None
            else:
                self.client = AsyncOpenAI(
                    base_url=self.api_url,
                    api_key=self.api_key,
                    timeout=self.api_timeout
                )
                logger.info(f"AI分析器初始化成功，模型: {self.api_model}")
                
        except Exception as e:
            logger.error(f"AI分析器初始化失败: {e}")
            self.client = None
    
    def _format_technical_data(self, df: pd.DataFrame) -> str:
        """
        格式化技术指标数据用于AI分析
        """
        if df.empty:
            return "无技术指标数据"
        
        try:
            latest = df.iloc[-1]
            
            # 移动平均线
            ma_data = []
            for period in [5, 10, 20, 60]:
                ma_key = f'MA{period}'
                if ma_key in latest:
                    ma_data.append(f"MA{period}: {latest[ma_key]:.2f}")
            
            # RSI
            rsi = latest.get('RSI', 'N/A')
            rsi_str = f"RSI(14): {rsi:.2f}" if pd.notna(rsi) else "RSI(14): N/A"
            
            # MACD
            macd = latest.get('MACD', 'N/A')
            signal = latest.get('Signal', 'N/A')
            macd_str = f"MACD: {macd:.4f}, Signal: {signal:.4f}" if pd.notna(macd) and pd.notna(signal) else "MACD: N/A"
            
            # 布林带
            bb_upper = latest.get('BB_Upper', 'N/A')
            bb_lower = latest.get('BB_Lower', 'N/A')
            bb_str = f"布林带上轨: {bb_upper:.2f}, 下轨: {bb_lower:.2f}" if pd.notna(bb_upper) and pd.notna(bb_lower) else "布林带: N/A"
            
            # 成交量
            volume_ratio = latest.get('Volume_Ratio', 'N/A')
            volume_str = f"成交量比率: {volume_ratio:.2f}" if pd.notna(volume_ratio) else "成交量比率: N/A"
            
            # 波动率
            volatility = latest.get('Volatility', 'N/A')
            vol_str = f"波动率: {volatility:.2f}%" if pd.notna(volatility) else "波动率: N/A"
            
            return f"""
技术指标分析:
移动平均线: {', '.join(ma_data)}
{rsi_str}
{macd_str}
{bb_str}
{volume_str}
{vol_str}
"""
        except Exception as e:
            logger.error(f"格式化技术数据时出错: {e}")
            return "技术指标数据格式化失败"
    
    def _format_price_data(self, df: pd.DataFrame) -> str:
        """
        格式化价格数据用于AI分析
        """
        if df.empty:
            return "无价格数据"
        
        try:
            # 获取最近5天的数据
            recent_data = df.tail(5)
            
            price_info = []
            for idx, row in recent_data.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                price_info.append(f"{date_str}: 开盘{row.get('Open', 'N/A'):.2f}, 收盘{row.get('Close', 'N/A'):.2f}, 最高{row.get('High', 'N/A'):.2f}, 最低{row.get('Low', 'N/A'):.2f}")
            
            return f"最近5天价格数据:\n" + "\n".join(price_info)
            
        except Exception as e:
            logger.error(f"格式化价格数据时出错: {e}")
            return "价格数据格式化失败"
    
    async def get_ai_analysis(self, df: pd.DataFrame, stock_code: str, 
                            market_type: str = 'A', stream: bool = False) -> AsyncGenerator[str, None]:
        """
        获取AI分析结果
        
        Args:
            df: 包含技术指标的DataFrame
            stock_code: 股票代码
            market_type: 市场类型
            stream: 是否使用流式响应
            
        Returns:
            异步生成器，生成AI分析结果
        """
        if not self.client:
            yield json.dumps({
                "error": "AI客户端未初始化",
                "ai_analysis": "无法进行AI分析，请检查API配置"
            })
            return
        
        try:
            # 格式化数据
            technical_data = self._format_technical_data(df)
            price_data = self._format_price_data(df)
            
            # 构建提示词
            market_names = {'A': 'A股', 'HK': '港股', 'US': '美股'}
            market_name = market_names.get(market_type, market_type)
            
            prompt = f"""
请对以下{market_name}股票进行专业的投资分析：

股票代码: {stock_code}

{technical_data}

{price_data}

请从以下角度进行分析：
1. **技术面分析**: 基于技术指标分析当前趋势和可能的走势
2. **风险提示**: 指出当前面临的主要风险
3. **投资建议**: 给出具体的投资建议（买入/持有/卖出）
4. **操作策略**: 建议的操作策略和注意事项

请用中文回答，格式要清晰易读。
"""
            
            # 调用AI模型，添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.api_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=stream,
                        max_tokens=1000,
                        temperature=0.7,
                        timeout=60  # 设置60秒超时
                    )
                    
                    if stream:
                        # 流式响应
                        full_response = ""
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                yield json.dumps({
                                    "ai_analysis_chunk": content,
                                    "stream_type": "ai_analysis"
                                })
                        
                        # 发送完整结果
                        yield json.dumps({
                            "ai_analysis": full_response,
                            "stream_type": "ai_analysis_complete"
                        })
                    else:
                        # 非流式响应
                        ai_analysis = response.choices[0].message.content
                        yield json.dumps({
                            "ai_analysis": ai_analysis,
                            "stream_type": "ai_analysis_complete"
                        })
                    
                    # 如果成功，跳出重试循环
                    break
                    
                except Exception as e:
                    logger.warning(f"AI分析第{attempt + 1}次尝试失败: {str(e)}")
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，抛出异常
                        raise e
                    else:
                        # 等待一段时间后重试
                        import asyncio
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"AI分析出错: {error_msg}")
            
            # 根据错误类型提供不同的错误信息
            if "deployment failed" in error_msg.lower() or "serving the request" in error_msg.lower():
                detailed_error = "AI模型部署失败，可能是模型服务暂时不可用。请稍后重试或检查API配置。"
            elif "timeout" in error_msg.lower():
                detailed_error = "AI分析请求超时，可能是网络问题或模型响应较慢。请稍后重试。"
            elif "rate limit" in error_msg.lower():
                detailed_error = "API调用频率超限，请稍后重试。"
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                detailed_error = "API认证失败，请检查API密钥是否正确。"
            else:
                detailed_error = f"AI分析失败: {error_msg}"
            
            yield json.dumps({
                "error": detailed_error,
                "ai_analysis": f"AI分析暂时不可用。错误详情: {error_msg}"
            })
    
    async def get_simple_analysis(self, stock_code: str, market_type: str = 'A') -> str:
        """
        获取简单的AI分析（用于快速分析）
        """
        if not self.client:
            return "AI客户端未初始化"
        
        try:
            market_names = {'A': 'A股', 'HK': '港股', 'US': '美股'}
            market_name = market_names.get(market_type, market_type)
            
            prompt = f"""
请简要分析{market_name}股票 {stock_code} 的投资价值，给出买入/持有/卖出的建议。
请用中文回答，控制在100字以内。
"""
            
            # 添加重试机制
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.api_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.5,
                        timeout=30  # 设置30秒超时
                    )
                    
                    return response.choices[0].message.content
                    
                except Exception as e:
                    logger.warning(f"简单AI分析第{attempt + 1}次尝试失败: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    else:
                        import asyncio
                        await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"简单AI分析出错: {error_msg}")
            
            if "deployment failed" in error_msg.lower():
                return "AI模型部署失败，请稍后重试"
            elif "timeout" in error_msg.lower():
                return "AI分析超时，请稍后重试"
            else:
                return f"AI分析失败: {error_msg}" 