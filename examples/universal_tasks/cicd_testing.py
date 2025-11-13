#!/usr/bin/env python3
"""
Universal Task CICD Testing Script

批量测试不同机械臂和不同任务的执行结果
生成详细的测试报告和统计信息
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass, asdict

import discoverse
from discoverse import DISCOVERSE_ROOT_DIR


@dataclass
class TestResult:
    """测试结果数据类"""
    robot_name: str
    task_name: str
    success: bool
    execution_time: float
    error_message: str = ""
    states_completed: int = 0
    total_states: int = 0
    timestamp: str = ""


class UniversalTaskCICD:
    """通用任务CICD测试器"""
    
    def __init__(self, output_dir: str = None):
        """初始化测试器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir or os.path.join(DISCOVERSE_ROOT_DIR, "cicd_reports")
        self.ensure_output_dir()
        
        # 支持的机械臂和任务
        self.supported_robots = [
            "airbot_play", "panda", "ur5e", "iiwa14", 
            "arx_x5", "arx_l5", "piper", "rm65", "xarm7"
        ]
        
        self.supported_tasks = [
            "place_block", "cover_cup", "stack_block", 
            "place_kiwi_fruit", "place_coffeecup"
        ]
        
        # 测试结果
        self.test_results: List[TestResult] = []
        
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_single_test(self, robot_name: str, task_name: str, timeout: int = 60) -> TestResult:
        """运行单个测试
        
        Args:
            robot_name: 机械臂名称
            task_name: 任务名称
            timeout: 超时时间（秒）
            
        Returns:
            测试结果
        """        
        # 构建命令
        script_path = os.path.join(
            DISCOVERSE_ROOT_DIR, 
            "discoverse/examples/universal_tasks/universal_task_runtime.py"
        )
        
        cmd = [
            sys.executable, script_path,
            "-r", robot_name,
            "-t", task_name,
            "-1",  # 单次执行
            "--headless"  # 无GUI模式
        ]
        
        start_time = time.time()
        result = TestResult(
            robot_name=robot_name,
            task_name=task_name,
            success=False,
            execution_time=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # 执行测试
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=DISCOVERSE_ROOT_DIR
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # 解析输出
            output = process.stdout
            stderr = process.stderr
            
            print(f"🧪 测试: {robot_name} - {task_name}")
            # 检查是否成功
            if "✅ 任务成功检查通过" in output or "🎉" in output and "任务成功完成" in output:
                result.success = True
                print(f"   ✅ 成功 ({execution_time:.2f}s)")
            else:
                result.success = False
                result.error_message = self._extract_error_message(output, stderr)
                print(f"   ❌ 失败 ({execution_time:.2f}s): {result.error_message[:100]}")
            
            # 提取状态信息
            result.states_completed, result.total_states = self._extract_state_info(output)
            
        except subprocess.TimeoutExpired:
            result.execution_time = timeout
            result.error_message = f"Timeout after {timeout}s"
            print(f"   ⏰ 超时 ({timeout}s)")
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_message = str(e)
            print(f"   💥 异常: {e}")
            
        return result
    
    def _extract_error_message(self, stdout: str, stderr: str) -> str:
        """提取错误信息"""
        # 从输出中提取关键错误信息
        error_keywords = [
            "❌", "失败", "Failed", "Error", "Exception", 
            "Traceback", "任务创建失败", "IK求解失败"
        ]
        
        lines = (stdout + "\n" + stderr).split('\n')
        error_lines = []
        
        for line in lines:
            if any(keyword in line for keyword in error_keywords):
                error_lines.append(line.strip())
                
        return " | ".join(error_lines[-3:])  # 只保留最后3个错误行
    
    def _extract_state_info(self, output: str) -> Tuple[int, int]:
        """提取状态完成信息"""
        import re
        
        # 查找状态完成信息
        state_pattern = r"完成状态: (\d+)/(\d+)"
        match = re.search(state_pattern, output)
        
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # 备选方案：计算状态数量
        state_count_pattern = r"🎯 状态 (\d+)/(\d+):"
        matches = re.findall(state_count_pattern, output)
        
        if matches:
            last_match = matches[-1]
            return int(last_match[0]), int(last_match[1])
            
        return 0, 0
    
    def run_batch_tests(self, 
                       robots: List[str] = None, 
                       tasks: List[str] = None,
                       parallel: bool = True,
                       max_workers: int = 4) -> Dict[str, Any]:
        """运行批量测试
        
        Args:
            robots: 要测试的机械臂列表
            tasks: 要测试的任务列表
            parallel: 是否并行执行
            max_workers: 最大并行工作数
            
        Returns:
            测试统计信息
        """
        robots = robots or self.supported_robots
        tasks = tasks or self.supported_tasks
        
        print(f"🚀 开始批量测试")
        print(f"   机械臂: {robots}")
        print(f"   任务: {tasks}")
        print(f"   并行: {parallel} (max_workers={max_workers})")
        print(f"   总计: {len(robots)} × {len(tasks)} = {len(robots) * len(tasks)} 个测试")
        print("=" * 70)
        
        start_time = time.time()
        
        # 生成测试用例
        test_cases = [(robot, task) for robot in robots for task in tasks]
        
        if parallel:
            # 并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_case = {
                    executor.submit(self.run_single_test, robot, task): (robot, task)
                    for robot, task in test_cases
                }
                
                for future in concurrent.futures.as_completed(future_to_case):
                    result = future.result()
                    self.test_results.append(result)
        else:
            # 串行执行
            for robot, task in test_cases:
                result = self.run_single_test(robot, task)
                self.test_results.append(result)
        
        total_time = time.time() - start_time
        
        # 生成统计信息
        stats = self._generate_statistics(total_time)
        
        # 保存结果
        self._save_results()
        self._save_statistics(stats)
        
        return stats
    
    def _generate_statistics(self, total_time: float) -> Dict[str, Any]:
        """生成测试统计信息"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        # 按机械臂统计
        robot_stats = {}
        for robot in self.supported_robots:
            robot_results = [r for r in self.test_results if r.robot_name == robot]
            robot_stats[robot] = {
                "total": len(robot_results),
                "success": sum(1 for r in robot_results if r.success),
                "failure": sum(1 for r in robot_results if not r.success),
                "success_rate": sum(1 for r in robot_results if r.success) / len(robot_results) if robot_results else 0
            }
        
        # 按任务统计
        task_stats = {}
        for task in self.supported_tasks:
            task_results = [r for r in self.test_results if r.task_name == task]
            task_stats[task] = {
                "total": len(task_results),
                "success": sum(1 for r in task_results if r.success),
                "failure": sum(1 for r in task_results if not r.success),
                "success_rate": sum(1 for r in task_results if r.success) / len(task_results) if task_results else 0
            }
        
        # 性能统计
        execution_times = [r.execution_time for r in self.test_results if r.success]
        
        stats = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time
            },
            "robot_statistics": robot_stats,
            "task_statistics": task_stats,
            "performance": {
                "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0
            },
            "failures": [
                {
                    "robot": r.robot_name,
                    "task": r.task_name,
                    "error": r.error_message,
                    "states": f"{r.states_completed}/{r.total_states}"
                }
                for r in self.test_results if not r.success
            ]
        }
        
        return stats
    
    def _save_results(self):
        """保存详细测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"test_results_{timestamp}.json")
        
        results_data = [asdict(result) for result in self.test_results]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"📄 详细结果保存至: {results_file}")
    
    def _save_statistics(self, stats: Dict[str, Any]):
        """保存统计信息"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(self.output_dir, f"test_statistics_{timestamp}.json")
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print(f"📊 统计信息保存至: {stats_file}")
        
        # 也保存为最新的统计文件
        latest_stats_file = os.path.join(self.output_dir, "latest_statistics.json")
        with open(latest_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, stats: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "=" * 70)
        print("📊 测试摘要")
        print("=" * 70)
        
        summary = stats["summary"]
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功: {summary['successful_tests']} ({summary['success_rate']:.1%})")
        print(f"失败: {summary['failed_tests']}")
        print(f"总耗时: {summary['total_execution_time']:.2f}s")
        
        print("\n🤖 机械臂成功率:")
        for robot, stat in stats["robot_statistics"].items():
            if stat["total"] > 0:
                print(f"  {robot}: {stat['success']}/{stat['total']} ({stat['success_rate']:.1%})")
        
        print("\n📋 任务成功率:")
        for task, stat in stats["task_statistics"].items():
            if stat["total"] > 0:
                print(f"  {task}: {stat['success']}/{stat['total']} ({stat['success_rate']:.1%})")
        
        if stats["failures"]:
            print(f"\n❌ 失败详情 ({len(stats['failures'])} 个):")
            for failure in stats["failures"][:5]:  # 只显示前5个
                print(f"  {failure['robot']}-{failure['task']}: {failure['error'][:60]}...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Universal Task CICD Testing")
    parser.add_argument("-r", "--robots", nargs="+", 
                       help="指定要测试的机械臂 (默认: 全部)")
    parser.add_argument("-t", "--tasks", nargs="+",
                       help="指定要测试的任务 (默认: 全部)")
    parser.add_argument("-o", "--output", type=str,
                       help="输出目录")
    parser.add_argument("--serial", action="store_true",
                       help="串行执行测试 (默认: 并行)")
    parser.add_argument("--workers", type=int, default=9,
                       help="并行工作数 (默认: 4)")
    parser.add_argument("--timeout", type=int, default=120,
                       help="单个测试超时时间/秒 (默认: 120)")
    
    args = parser.parse_args()
    
    # 创建测试器
    cicd = UniversalTaskCICD(output_dir=args.output)
    
    # 运行测试
    stats = cicd.run_batch_tests(
        robots=args.robots,
        tasks=args.tasks,
        parallel=not args.serial,
        max_workers=args.workers
    )
    
    # 打印摘要
    cicd.print_summary(stats)


if __name__ == "__main__":
    main()
