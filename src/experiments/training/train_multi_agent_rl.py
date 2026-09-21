"""
多智能体RL训练 - 同时训练多个智能体互相竞争
采用循环训练策略：每次训练一个智能体，其他智能体使用最新模型
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from train_rl_enhanced import EnhancedMarketEnv
from agents.zi_agent import ZIAgent
from agents.rl_agent_enhanced import EnhancedRLAgent


class MultiAgentTrainingCallback(BaseCallback):
    """多智能体训练回调"""
    def __init__(self, training_agent_id: str, verbose=0):
        super().__init__(verbose)
        self.training_agent_id = training_agent_id
        self.episode_rewards = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.episode_rewards.append(mean_reward)
            self.episode_count = len(self.episode_rewards)


def select_training_agents(agents_config: List[Dict], num_agents: int = 8) -> List[Dict]:
    """
    选择代表性的智能体进行训练
    选择策略：覆盖不同燃料类型和成本区间
    """
    # 按燃料类型分组
    fuel_groups = {}
    for agent in agents_config:
        fuel = agent['config'].get('fuel_category', 'Unknown')
        if fuel not in fuel_groups:
            fuel_groups[fuel] = []
        fuel_groups[fuel].append(agent)
    
    selected = []
    
    # 从每个燃料类型中选择代表
    for fuel_type, agents in fuel_groups.items():
        # 按成本排序
        agents_sorted = sorted(agents, key=lambda x: x['config'].get('marginal_cost_energy', 50))
        
        # 选择低成本和高成本各一个
        if len(agents_sorted) >= 2:
            selected.append(agents_sorted[0])  # 最低成本
            selected.append(agents_sorted[-1])  # 最高成本
        elif len(agents_sorted) == 1:
            selected.append(agents_sorted[0])
    
    # 如果不够，补充中等容量的
    if len(selected) < num_agents:
        remaining = [a for a in agents_config if a not in selected]
        remaining_sorted = sorted(remaining, key=lambda x: x['config'].get('max_capacity', 0), reverse=True)
        selected.extend(remaining_sorted[:num_agents - len(selected)])
    
    return selected[:num_agents]


def train_multi_agent_rl(
    config: Dict,
    training_agents: List[Dict],
    total_timesteps_per_agent: int = 2500,
    num_rounds: int = 50,
    save_dir: str = "models/multi_agent_high_price"
):
    """
    多智能体循环训练
    
    Args:
        config: 仿真配置
        training_agents: 要训练的智能体列表
        total_timesteps_per_agent: 每个智能体每轮训练步数
        num_rounds: 训练轮数（每轮所有智能体都训练一次）
        save_dir: 模型保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化：为每个训练智能体创建模型
    models = {}
    for agent_data in training_agents:
        agent_id = agent_data['agent_id']
        model_path = os.path.join(save_dir, f"{agent_id}_model")
        
        # 创建环境
        env = EnhancedMarketEnv(
            config=config,
            training_agent_id=agent_id,
            opponent_agents={}  # 稍后填充
        )
        
        # 创建或加载模型
        if os.path.exists(f"{model_path}.zip"):
            print(f"   加载已有模型: {agent_id}")
            models[agent_id] = PPO.load(model_path, env=env)
        else:
            print(f"   创建新模型: {agent_id}")
            models[agent_id] = PPO(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
                n_steps=256,
                batch_size=128,
                n_epochs=5,
                gamma=0.99,
                verbose=0
            )
    
    print(f"\n✅ 已初始化 {len(models)} 个智能体模型\n")
    
    # 获取所有智能体配置
    all_agents_config = config['agents']
    all_agents_info = [a['config'] for a in all_agents_config]
    
    # 循环训练
    for round_idx in range(num_rounds):
        print("="*80)
        print(f"🔄 训练轮次 {round_idx + 1}/{num_rounds}")
        print("="*80)
        
        for train_idx, agent_data in enumerate(training_agents):
            agent_id = agent_data['agent_id']
            agent_config = agent_data['config']
            
            print(f"\n📍 [{train_idx + 1}/{len(training_agents)}] 训练智能体: {agent_id}")
            print(f"   燃料类型: {agent_config.get('fuel_category', 'N/A')}")
            print(f"   成本: {agent_config.get('marginal_cost_energy', 0):.2f} 元/MWh")
            print(f"   容量: {agent_config.get('max_capacity', 0):.0f} MW")
            
            # 创建对手智能体
            opponent_agents = {}
            
            # 1. 其他训练中的智能体使用最新模型
            for other_agent_data in training_agents:
                other_id = other_agent_data['agent_id']
                if other_id != agent_id:
                    other_config = other_agent_data['config']
                    model_path = os.path.join(save_dir, f"{other_id}_model.zip")
                    
                    if os.path.exists(model_path):
                        opponent_agents[other_id] = EnhancedRLAgent(
                            agent_id=other_id,
                            config=other_config,
                            model_path=model_path,
                            all_agents_info=all_agents_info
                        )
                    else:
                        # 如果还没有模型，使用ZI
                        opponent_agents[other_id] = ZIAgent(other_id, other_config)
            
            # 2. 其余智能体使用ZI
            for other_agent_data in all_agents_config:
                other_id = other_agent_data['agent_id']
                if other_id not in opponent_agents and other_id != agent_id:
                    opponent_agents[other_id] = ZIAgent(
                        other_id, other_agent_data['config']
                    )
            
            print(f"   对手: {len([a for a in opponent_agents.values() if isinstance(a, EnhancedRLAgent)])} 个RL + {len([a for a in opponent_agents.values() if isinstance(a, ZIAgent)])} 个ZI")
            
            # 创建训练环境
            env = EnhancedMarketEnv(
                config=config,
                training_agent_id=agent_id,
                opponent_agents=opponent_agents
            )
            
            # 更新模型的环境
            models[agent_id].set_env(env)
            
            # 训练
            callback = MultiAgentTrainingCallback(agent_id)
            models[agent_id].learn(
                total_timesteps=total_timesteps_per_agent,
                callback=callback,
                reset_num_timesteps=True,  # ✅ 修复：每次重置计数器，避免重复训练
                progress_bar=True
            )
            
            # 保存模型
            model_path = os.path.join(save_dir, f"{agent_id}_model")
            models[agent_id].save(model_path)
            print(f"   ✅ 模型已保存: {model_path}.zip")
            
            # 显示训练统计
            if len(callback.episode_rewards) > 0:
                print(f"   平均奖励: {np.mean(callback.episode_rewards[-10:]):.2f}")
        
        print(f"\n✅ 第 {round_idx + 1} 轮训练完成\n")
    
    return models


def main():
    print("="*80)
    print("🚀 多智能体RL训练 - 互相竞争学习")
    print("="*80)
    
    # 1. 加载配置
    print("\n📋 步骤1：加载配置")
    agents_file = "true_digital_twin_agents.json"
    with open(agents_file, 'r', encoding='utf-8') as f:
        agents_config = json.load(f)
    
    # 创建基础配置
    config = {
        "num_rounds": 10,
        "agents": agents_config,
        "carbon_market": {
            "enabled": True,
            "init_carbon_price": 143.0,
            "min_carbon_price": 113.0
        },
        "logging": {
            "log_file": "logs/multi_agent_training.log",
            "results_file_csv": "results/multi_agent_training_results.csv"
        }
    }
    
    # 加载碳配额配置
    try:
        with open('tiered_carbon_quota_config.json', 'r', encoding='utf-8') as f:
            quota_config = json.load(f)
        config['carbon_market']['tiered_quota'] = quota_config['tiered_quota_system']
    except:
        pass
    
    config['agents'] = agents_config
    config['num_rounds'] = 10
    
    print(f"   ✅ 已加载 {len(agents_config)} 个智能体配置")
    
    # 2. 选择训练智能体
    print("\n🎯 步骤2：选择代表性智能体进行训练")
    num_training_agents = 8  # 同时训练8个智能体
    training_agents = select_training_agents(agents_config, num_training_agents)
    
    print(f"   选中 {len(training_agents)} 个智能体:")
    for agent in training_agents:
        fuel = agent['config'].get('fuel_category', 'N/A')
        cost = agent['config'].get('marginal_cost_energy', 0)
        capacity = agent['config'].get('max_capacity', 0)
        print(f"   - {agent['agent_id']}: {fuel}, 成本={cost:.2f}, 容量={capacity:.0f}MW")
    
    # 3. 开始多智能体训练
    print("\n🏋️  步骤3：开始多智能体循环训练")
    print("="*80)
    print(f"   训练智能体数: {len(training_agents)}")
    print(f"   每个智能体训练步数: 2,500步/轮")
    print(f"   训练轮数: 50轮")
    print(f"   总训练步数: {len(training_agents) * 2500 * 50:,}步")
    print("   策略: 循环训练，智能体之间互相竞争")
    print("="*80)
    
    models = train_multi_agent_rl(
        config=config,
        training_agents=training_agents,
        total_timesteps_per_agent=2500,
        num_rounds=50,
        save_dir="models/multi_agent_high_price"
    )
    
    print("\n" + "="*80)
    print("✅ 多智能体训练完成！")
    print("="*80)
    print(f"   训练完成的智能体数: {len(models)}")
    print(f"   模型保存位置: models/multi_agent/")
    print("\n💡 下一步:")
    print("   运行: python run_multi_agent_simulation.py")
    print("="*80)


if __name__ == "__main__":
    main()
