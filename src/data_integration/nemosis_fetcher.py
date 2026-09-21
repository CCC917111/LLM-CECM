#!/usr/bin/env python3
"""
使用NEMOSIS获取真实AEMO数据

NEMOSIS是UNSW开发的专业AEMO数据获取工具包
文档: https://github.com/UNSW-CEEM/NEMOSIS
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json

try:
    from nemosis import dynamic_data_compiler
    NEMOSIS_AVAILABLE = True
except ImportError:
    NEMOSIS_AVAILABLE = False
    print("⚠️ NEMOSIS未安装，请运行: pip install nemosis")


class RealAEMODataClient:
    """使用NEMOSIS获取真实AEMO数据的客户端"""
    
    def __init__(self, cache_dir="data_cache/nemosis"):
        """
        初始化数据客户端
        
        Args:
            cache_dir: NEMOSIS数据缓存目录
        """
        if not NEMOSIS_AVAILABLE:
            raise ImportError("请先安装nemosis: pip install nemosis")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_dispatch_prices(self, start_date: str, end_date: str, 
                           region: str = 'NSW1') -> pd.DataFrame:
        """
        获取调度电价（5分钟间隔）
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            region: 地区 NSW1/VIC1/QLD1/SA1/TAS1
            
        Returns:
            包含时间和价格的DataFrame
        """
        self.logger.info(f"获取 {region} 地区 {start_date} 至 {end_date} 的电价数据...")
        
        # 转换日期格式为NEMOSIS要求的格式
        start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y/%m/%d %H:%M:%S')
        end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y/%m/%d %H:%M:%S')
        
        try:
            # 使用NEMOSIS获取调度价格数据
            prices_df = dynamic_data_compiler(
                start_time=start,
                end_time=end,
                table_name='DISPATCHPRICE',  # 5分钟调度价格
                raw_data_location=str(self.cache_dir)
            )
            
            # 筛选特定地区
            if 'REGIONID' in prices_df.columns:
                prices_df = prices_df[prices_df['REGIONID'] == region].copy()
            
            # 确保有时间列
            if 'SETTLEMENTDATE' in prices_df.columns:
                prices_df['SETTLEMENTDATE'] = pd.to_datetime(prices_df['SETTLEMENTDATE'])
            
            self.logger.info(f"✓ 获取到 {len(prices_df)} 条价格记录")
            return prices_df[['SETTLEMENTDATE', 'REGIONID', 'RRP']].sort_values('SETTLEMENTDATE')
            
        except Exception as e:
            self.logger.error(f"获取价格数据失败: {e}")
            raise
    
    def get_generator_dispatch(self, start_date: str, end_date: str, 
                              duids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取机组调度出力数据
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            duids: 机组ID列表（可选，为None则获取所有）
            
        Returns:
            包含机组出力的DataFrame
        """
        self.logger.info(f"获取 {start_date} 至 {end_date} 的机组调度数据...")
        
        start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y/%m/%d %H:%M:%S')
        end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y/%m/%d %H:%M:%S')
        
        try:
            # 获取机组调度数据
            dispatch_df = dynamic_data_compiler(
                start_time=start,
                end_time=end,
                table_name='DISPATCHLOAD',  # 机组调度出力
                raw_data_location=str(self.cache_dir)
            )
            
            # 筛选特定机组
            if duids and 'DUID' in dispatch_df.columns:
                dispatch_df = dispatch_df[dispatch_df['DUID'].isin(duids)].copy()
            
            if 'SETTLEMENTDATE' in dispatch_df.columns:
                dispatch_df['SETTLEMENTDATE'] = pd.to_datetime(dispatch_df['SETTLEMENTDATE'])
            
            self.logger.info(f"✓ 获取到 {len(dispatch_df)} 条调度记录")
            return dispatch_df.sort_values('SETTLEMENTDATE')
            
        except Exception as e:
            self.logger.error(f"获取调度数据失败: {e}")
            raise
    
    def get_aggregated_hourly_prices(self, start_date: str, end_date: str,
                                    region: str = 'NSW1') -> List[float]:
        """
        获取小时级聚合电价（用于实验配置）
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'  
            region: 地区代码
            
        Returns:
            小时均价列表
        """
        # 获取5分钟数据
        prices_df = self.get_dispatch_prices(start_date, end_date, region)
        
        # 重采样为小时均价
        prices_df.set_index('SETTLEMENTDATE', inplace=True)
        hourly_prices = prices_df['RRP'].resample('1H').mean()
        
        return hourly_prices.tolist()


def create_experiment_config_with_real_data(
    start_date: str = '2024-07-15',
    num_days: int = 1,
    region: str = 'NSW1'
) -> Dict:
    """
    基于真实AEMO数据创建实验配置
    
    Args:
        start_date: 开始日期
        num_days: 天数
        region: 地区
        
    Returns:
        更新后的实验配置字典
    """
    if not NEMOSIS_AVAILABLE:
        raise ImportError("请先安装nemosis: pip install nemosis")
    
    client = RealAEMODataClient()
    
    # 计算日期范围
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=num_days)
    end_date = end.strftime('%Y-%m-%d')
    
    print(f"正在获取 {start_date} 至 {end_date} 的真实AEMO数据...")
    
    # 获取小时电价数据
    hourly_prices = client.get_aggregated_hourly_prices(start_date, end_date, region)
    
    # 取前10个小时用于实验（对应10轮）
    ground_truth_rrp = hourly_prices[:10]
    
    print(f"✓ 获取到 {len(ground_truth_rrp)} 小时的电价数据")
    print(f"  价格范围: ${min(ground_truth_rrp):.2f} - ${max(ground_truth_rrp):.2f}")
    
    # 构建配置更新
    real_data_config = {
        'real_data_dynamics': {
            'enabled': True,
            'ground_truth_rrp': ground_truth_rrp,
            'data_source': f'AEMO NEMWEB - {region} {start_date}',
            'data_fetched_at': datetime.now().isoformat(),
        }
    }
    
    return real_data_config


def download_sample_data():
    """下载示例数据供测试使用"""
    if not NEMOSIS_AVAILABLE:
        print("❌ 请先安装nemosis: pip install nemosis")
        return
    
    print("="*80)
    print("下载真实AEMO数据示例")
    print("="*80)
    
    client = RealAEMODataClient()
    
    # 示例：2024年7月15日的数据
    test_date = '2024-07-15'
    
    try:
        # 1. 获取电价
        print(f"\n1. 获取 {test_date} 的电价数据...")
        prices = client.get_dispatch_prices(test_date, test_date, 'NSW1')
        print(f"   ✓ 成功获取 {len(prices)} 条价格记录")
        
        # 保存样本
        prices_sample = prices.head(20)
        print("\n   前20条价格数据：")
        print(prices_sample[['SETTLEMENTDATE', 'RRP']].to_string(index=False))
        
        # 2. 获取小时聚合数据
        print(f"\n2. 计算小时均价...")
        hourly = client.get_aggregated_hourly_prices(test_date, test_date, 'NSW1')
        print(f"   ✓ 获取到 {len(hourly)} 小时数据")
        print(f"   小时均价: {hourly}")
        
        # 3. 保存配置
        print(f"\n3. 生成实验配置...")
        config = create_experiment_config_with_real_data(test_date)
        
        output_file = "real_aemo_data_config.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ 配置已保存到: {output_file}")
        
        print("\n" + "="*80)
        print("✅ 数据下载完成！")
        print("="*80)
        print("\n💡 提示:")
        print("  - 数据已缓存到 data_cache/nemosis/")
        print("  - 后续访问相同日期数据将直接使用缓存")
        print("  - 可在实验配置中使用 real_aemo_data_config.json 中的数据")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("  - 网络连接问题")
        print("  - NEMWEB服务器暂时不可用")
        print("  - 指定日期的数据不存在")
        print("\n建议:")
        print("  - 检查网络连接")
        print("  - 尝试不同的日期（建议使用1-2年内的数据）")


if __name__ == "__main__":
    download_sample_data()
