"""
LLM客户端模块
封装与大语言模型的交互
"""

import os
import re
import time
import json
import random
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from simulation.config import LLM_MODEL_NAME

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 通过 Google Generative Language REST API 调用 Gemini / Gemma 系列模型（见 LLMClient.chat）

# 可选：从项目根目录的 .env 文件加载环境变量（需要 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_api_keys_from_env() -> List[str]:
    """
    从环境变量 GOOGLE_API_KEY 读取API密钥。
    多个密钥用英文逗号分隔，客户端会轮询使用以分摊请求配额，例如：
        export GOOGLE_API_KEY="key1,key2"
    """
    raw = os.environ.get('GOOGLE_API_KEY', '')
    return [key.strip() for key in raw.split(',') if key.strip()]


class LLMClient:
    """LLM客户端类，封装与大语言模型的交互"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化LLM客户端
        
        Args:
            config: 配置参数字典，可包含以下键：
                - api_keys: API密钥列表（不提供时从环境变量 GOOGLE_API_KEY 读取）
                - max_retries: 最大重试次数
                - request_timeout: 请求超时时间（秒）
                - random_seed: 随机种子
                - offline: 离线模式（不发送网络请求，用于测试）
        """
        if config is None:
            config = {}
            
        # 设置离线模式（用于无网络或加速本地仿真）
        self.offline = config.get('offline', False)

        # 设置重试和超时参数
        self.max_retries = config.get('max_retries', 5)
        self.request_timeout = config.get('request_timeout', 120)
        self.random_seed = config.get('random_seed', None)
        
        # API密钥：优先使用配置中的密钥，否则读取环境变量
        self.api_keys = list(config.get('api_keys') or get_api_keys_from_env())
        if not self.api_keys and not self.offline:
            raise ValueError(
                "未找到LLM API密钥。请设置环境变量 GOOGLE_API_KEY（多个密钥用逗号分隔），"
                "或在项目根目录的 .env 文件中配置，参考 .env.example。"
            )

        # 添加轮询索引，用于实现round-robin API密钥选择
        self.current_key_index = 0
        # 添加计数器，记录每个API密钥的使用次数
        self.api_key_usage_counter = {key: 0 for key in self.api_keys}
        
        # 如果设置了随机种子，则固定随机数生成器
        if self.random_seed is not None:
            random.seed(self.random_seed)
            logging.info(f"LLM客户端已设置随机种子: {self.random_seed}")

        logging.info(f"LLMClient 初始化完成，可用密钥 {len(self.api_keys)} 个{'（离线模式）' if self.offline else ''}。")

    def chat(self, prompt: str, return_json: bool = True, model: str = LLM_MODEL_NAME, temperature: float = 0.3, track_tokens: bool = False) -> any:
        """
        向LLM发送请求并获取回复

        Args:
            prompt: 提示词
            return_json: 是否尝试将回复解析为JSON
            model: 使用的模型名称
            temperature: 温度参数(0-1)，越高越随机
            track_tokens: 是否返回token使用情况

        Returns:
            如果return_json为True，尝试返回JSON对象；否则返回文本
            如果track_tokens为True，返回(response, token_info)元组
        """
        # 离线模式：直接返回一个默认响应，避免网络调用
        if self.offline:
            default = {"default_response": True, "error": "offline mode"}
            if track_tokens:
                return default, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            return default

        # 添加延迟以避免API调用限制
        delay = 0
        if len(self.api_keys) > 1:
            delay = random.uniform(1.0, 3.0)  # 减少延迟时间，避免测试等待过长
        time.sleep(delay)
        
        # 使用轮询方式选择API密钥
        api_key = self.api_keys[self.current_key_index]
        # 更新轮询索引
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        # 记录API密钥使用次数
        self.api_key_usage_counter[api_key] = self.api_key_usage_counter.get(api_key, 0) + 1
        
        # 每50次API调用输出一次使用情况统计
        if sum(self.api_key_usage_counter.values()) % 50 == 0:
            total_calls = sum(self.api_key_usage_counter.values())
            usage_stats = "\n".join([f"  • {key[-8:]}: {count} 次 ({count/total_calls*100:.1f}%)" 
                                 for key, count in self.api_key_usage_counter.items() if count > 0])
            logging.info(f"API密钥使用统计 (总调用次数: {total_calls}):\n{usage_stats}")
            
        # 构建请求数据
        headers = {
            "Content-Type": "application/json",
        }

        # 根据模型名称调整请求格式
        if "gemini-2.0" in model:
            # Gemini 2.0 格式
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 64,
                    "maxOutputTokens": 16384,
                }
            }
        else:
            # 标准 Gemini 格式
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 64,
                    "maxOutputTokens": 16384,
                }
            }

        if self.random_seed is not None:
            data["generationConfig"]["seed"] = self.random_seed
            
        # 构建API URL
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        # 发送请求，带重试机制
        retries = 0
        while retries <= self.max_retries:
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout
                )
                # 记录原始响应内容，便于调试
                if response.status_code != 200:
                    logging.error(f"API响应错误: 状态码={response.status_code}, 响应内容={response.text}")
                response.raise_for_status()  # 如果状态码不是200，抛出异常
                result = response.json()
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                error_detail = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = f", 错误详情: {e.response.json()}"
                    except:
                        error_detail = f", 响应内容: {e.response.text[:500]}"
                
                if retries <= self.max_retries:
                    # 指数退避重试
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    logging.warning(f"API请求失败，{wait_time:.2f}秒后重试 ({retries}/{self.max_retries}): {str(e)}{error_detail}")
                    time.sleep(wait_time)
                    continue
                else:
                    # 其他错误或重试次数用完，抛出异常
                    logging.error(f"API请求最终失败: {str(e)}{error_detail}")
                    raise
            except Exception as e:
                # 非HTTP错误，直接抛出
                raise

        # 处理成功的响应
        try:
            # 提取token使用信息
            token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            
            # 优先从 Gemini 的 promptFeedback 和 candidates 中获取 token 计数
            if 'promptFeedback' in result and 'tokenCount' in result['promptFeedback']:
                token_info['prompt_tokens'] = result['promptFeedback']['tokenCount']
            
            if 'candidates' in result and len(result['candidates']) > 0 and 'tokenCount' in result['candidates'][0]:
                token_info['completion_tokens'] = result['candidates'][0]['tokenCount']
            
            token_info['total_tokens'] = token_info['prompt_tokens'] + token_info['completion_tokens']

            # 如果上述方式未能获取，则尝试原有的 usageMetadata 或 usage 键
            if token_info['total_tokens'] == 0:
                if 'usageMetadata' in result:
                    usage = result['usageMetadata']
                    token_info = {
                        'prompt_tokens': usage.get('promptTokenCount', 0),
                        'completion_tokens': usage.get('candidatesTokenCount', 0),
                        'total_tokens': usage.get('totalTokenCount', 0)
                    }
                elif 'usage' in result:
                    usage = result['usage']
                    token_info = {
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'completion_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    }
                else:
                    logging.debug("API响应中未找到token使用信息，记录为零")
            
            # 记录token信息
            logging.debug(f"API响应中的token信息: {token_info}")

            # 解析Gemini响应格式
            response_content = None
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                if return_json:
                    try:
                        # 尝试从文本中提取JSON
                        # 查找JSON代码块
                        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = content
                        response_content = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，返回原始内容
                        response_content = {"text": content}
                else:
                    response_content = content
            else:
                response_content = None

            # 根据track_tokens参数决定返回格式
            if track_tokens:
                return response_content, token_info
            else:
                return response_content

        except requests.exceptions.HTTPError as e:
            error_msg = f"Gemini API HTTP错误: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f", 详细信息: {error_detail}"
                except:
                    error_msg += f", 响应内容: {e.response.text[:500]}"
            print(error_msg)
            if track_tokens:
                return None, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            else:
                return None
        except Exception as e:
            print(f"Gemini API调用错误: {str(e)}")
            if track_tokens:
                return None, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            else:
                return None




# 示例用法 (可以放在 if __name__ == "__main__": 中测试)
if __name__ == "__main__":
    client = LLMClient()
    prompt_example = "你好，请用中文简单介绍一下你自己。"
    response_text = client.chat(prompt_example, return_json=False)
    print("文本响应:")
    print(response_text)

    prompt_json = "请以JSON格式返回一个包含'城市'和'天气'键的示例对象，城市是北京。"
    response_json = client.chat(prompt_json, return_json=True)
    print("\nJSON响应:")
    print(response_json)