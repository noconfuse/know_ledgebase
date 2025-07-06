"""自定义输出解析器，处理LLM输出的JSON格式问题"""

import json
import re
from typing import Optional, Type
from pydantic import BaseModel, ValidationError
from llama_index.core.output_parsers.base import ChainableOutputParser
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
import logging

logger = logging.getLogger(__name__)


class CustomOutputParser(ChainableOutputParser):
    """自定义输出解析器，处理LLM输出的JSON格式问题"""
    
    def __init__(self, output_cls: Type[BaseModel], verbose: bool = False):
        self.output_cls = output_cls
        self.verbose = verbose
        self.pydantic_parser = PydanticOutputParser(output_cls=output_cls)
    
    def parse(self, output: str):
        """解析LLM输出，处理properties包装等格式问题"""
        if self.verbose:
            logger.debug(f"Raw LLM output: {output}")
        
        try:
            # 尝试标准的Pydantic解析
            return self.pydantic_parser.parse(output)
        except ValidationError as e:
            logger.warning(f"Standard Pydantic validation failed: {e}")
            
            # 尝试处理LLM输出被properties包装的情况
            try:
                cleaned_output = self._handle_properties_wrapper(output)
                if cleaned_output != output:
                    logger.info("Detected and handled properties wrapper in LLM output")
                    return self.pydantic_parser.parse(cleaned_output)
            except Exception as cleanup_error:
                logger.debug(f"Properties wrapper handling failed: {cleanup_error}")
            
            # 尝试类型转换修复
            try:
                fixed_output = self._fix_type_mismatches(output, e)
                if fixed_output != output:
                    logger.info("Applied type conversion fixes to LLM output")
                    return self.pydantic_parser.parse(fixed_output)
            except Exception as type_fix_error:
                logger.debug(f"Type conversion fixes failed: {type_fix_error}")
            
            # 如果所有处理都失败，抛出原始ValidationError
            logger.error(f"All parsing attempts failed for output: {output[:200]}...")
            raise e
    
    def _handle_properties_wrapper(self, output: str) -> str:
        """处理LLM输出被properties结构包装的情况"""
        try:
            # 首先尝试解析为JSON
            parsed_json = json.loads(output)
            
            # 检查是否数据被包装在properties字段中
            if isinstance(parsed_json, dict) and "properties" in parsed_json:
                properties_content = parsed_json["properties"]
                if isinstance(properties_content, dict):
                    # 检查properties内容是否像实际数据而不是schema
                    if self._looks_like_data_not_schema(properties_content):
                        logger.info("Extracting data from properties wrapper")
                        return json.dumps(properties_content)
                    else:
                        logger.warning("Detected JSON schema format instead of data - LLM returned schema instead of actual values")
                        # 这种情况下，LLM返回了schema而不是数据，应该抛出错误
                        raise ValueError("LLM returned JSON schema instead of actual data")
            
            # 如果JSON解析成功但不是properties包装，返回原始输出
            return output
            
        except json.JSONDecodeError:
            # 如果不是有效JSON，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict) and "properties" in parsed_json:
                        properties_content = parsed_json["properties"]
                        if isinstance(properties_content, dict) and self._looks_like_data_not_schema(properties_content):
                            logger.info("Extracting data from properties wrapper in embedded JSON")
                            return json.dumps(properties_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed for extracted content: {e}")
                    # 尝试通过括号匹配来提取结构化字段
                    try:
                        cleaned_json = self._extract_json_by_brackets(json_str)
                        if cleaned_json:
                            parsed_json = json.loads(cleaned_json)
                            if isinstance(parsed_json, dict):
                                # 检查是否有properties包装
                                if "properties" in parsed_json:
                                    properties_content = parsed_json["properties"]
                                    if isinstance(properties_content, dict) and self._looks_like_data_not_schema(properties_content):
                                        logger.info("Successfully extracted data from properties wrapper using bracket matching")
                                        return json.dumps(properties_content)
                                else:
                                    # 直接返回解析的JSON
                                    logger.info("Successfully extracted JSON using bracket matching")
                                    return json.dumps(parsed_json)
                    except (json.JSONDecodeError, Exception) as bracket_error:
                        logger.warning(f"Bracket-based JSON extraction also failed: {bracket_error}")
            
            # 如果无法处理，返回原始输出
            return output
    
    def _looks_like_data_not_schema(self, content: dict) -> bool:
        """判断内容是否像实际数据而不是JSON schema"""
        # JSON schema通常包含type, description, title等字段
        schema_indicators = {"type", "description", "title", "items", "required", "enum"}
        
        for key, value in content.items():
            if isinstance(value, dict):
                # 如果值是字典且包含schema指示符，可能是schema
                if any(indicator in value for indicator in schema_indicators):
                    return False
            elif isinstance(value, str):
                # 如果值是字符串且不是schema描述，可能是实际数据
                continue
            elif isinstance(value, (list, int, float, bool)):
                # 如果值是基本数据类型，可能是实际数据
                continue
        
        return True
    
    def _fix_type_mismatches(self, output: str, validation_error: ValidationError) -> str:
        """修复类型不匹配问题
        
        Args:
            output: 原始LLM输出
            validation_error: Pydantic验证错误
            
        Returns:
            修复后的JSON字符串
        """
        try:
            # 解析原始JSON
            data = json.loads(output)
            if not isinstance(data, dict):
                return output
            
            # 分析验证错误，找出类型不匹配的字段
            modified = False
            for error in validation_error.errors():
                field_path = error.get('loc', [])
                error_type = error.get('type', '')
                input_value = error.get('input')
                
                if len(field_path) == 1 and error_type == 'list_type':
                    field_name = field_path[0]
                    if field_name in data:
                        current_value = data[field_name]
                        
                        # 如果当前值是字符串，但期望是列表
                        if isinstance(current_value, str):
                            # 尝试将字符串转换为单元素列表
                            data[field_name] = [current_value]
                            modified = True
                            logger.info(f"Converted string field '{field_name}' to list: '{current_value}' -> ['{current_value}']")
                        
                        # 如果当前值是其他类型，也尝试转换为列表
                        elif not isinstance(current_value, list):
                            data[field_name] = [str(current_value)]
                            modified = True
                            logger.info(f"Converted field '{field_name}' to list: {current_value} -> ['{current_value}']")
                
                elif len(field_path) == 1 and error_type == 'string_type':
                    field_name = field_path[0]
                    if field_name in data:
                        current_value = data[field_name]
                        
                        # 如果当前值是列表，但期望是字符串
                        if isinstance(current_value, list) and len(current_value) > 0:
                            # 取第一个元素作为字符串值
                            data[field_name] = str(current_value[0])
                            modified = True
                            logger.info(f"Converted list field '{field_name}' to string: {current_value} -> '{current_value[0]}'")
                        
                        # 如果当前值是其他类型，转换为字符串
                        elif not isinstance(current_value, str):
                            data[field_name] = str(current_value)
                            modified = True
                            logger.info(f"Converted field '{field_name}' to string: {current_value} -> '{current_value}'")
            
            if modified:
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                return output
                
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error in type mismatch fixing: {e}")
            return output
    
    def _extract_json_by_brackets(self, json_str: str) -> Optional[str]:
        """通过括号匹配来提取和清理JSON字符串
        
        Args:
            json_str: 可能包含格式问题的JSON字符串
            
        Returns:
            清理后的有效JSON字符串，如果无法修复则返回None
        """
        try:
            # 移除可能的前后空白和换行符
            json_str = json_str.strip()
            
            # 找到第一个 { 和最后一个 } 来确定JSON边界
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                return None
            
            # 提取JSON部分
            json_content = json_str[start_idx:end_idx + 1]
            
            # 尝试通过括号匹配来验证和修复JSON结构
            bracket_count = 0
            quote_count = 0
            in_string = False
            escape_next = False
            valid_end = -1
            
            for i, char in enumerate(json_content):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            valid_end = i
                            break
            
            if valid_end > 0:
                # 截取到有效的结束位置
                cleaned_json = json_content[:valid_end + 1]
                
                # 尝试一些常见的修复
                cleaned_json = self._fix_common_json_issues(cleaned_json)
                
                return cleaned_json
                
            return None
            
        except Exception as e:
            logger.warning(f"Error in bracket-based JSON extraction: {e}")
            return None
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """修复常见的JSON格式问题
        
        Args:
            json_str: 需要修复的JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        try:
            # 移除可能的尾随逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # 逐行处理字符串值中的未转义引号
            lines = json_str.split('\n')
            fixed_lines = []
            
            for line in lines:
                if ':' in line and '"' in line:
                    # 直接处理包含引号的字符串值
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0]
                        value_part = parts[1].strip()
                        
                        if value_part.startswith('"'):
                            # 处理字符串值
                            if value_part.count('"') > 2:  # 有未转义的引号
                                # 提取内容
                                if value_part.endswith(','):
                                    content = value_part[1:-2]
                                    trailing = ','
                                elif value_part.endswith('"'):
                                    content = value_part[1:-1]
                                    trailing = ''
                                else:
                                    fixed_lines.append(line)
                                    continue
                                
                                # 转义内部引号
                                escaped_content = content.replace('"', '\\"')
                                fixed_line = f'{key_part}: "{escaped_content}"{trailing}'
                                fixed_lines.append(fixed_line)
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            logger.warning(f"Error fixing JSON issues: {e}")
            return json_str
    
    def get_format_instructions(self) -> str:
        """获取格式说明"""
        return self.pydantic_parser.get_format_instructions()