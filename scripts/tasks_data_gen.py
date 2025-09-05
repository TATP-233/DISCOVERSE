import os
import argparse
import time
import psutil
import threading
import logging
import subprocess
from discoverse import DISCOVERSE_ROOT_DIR
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MemoryMonitor')

class ProcessInfo:
    def __init__(self, thread_id, future):
        self.pid = None  # 将由子进程PID填充
        self.thread_id = thread_id
        self.future = future
        self.memory_usage = 0  # 内存使用量(MB)
        self.subprocess = None  # 存储子进程对象
        
    def update_memory_usage(self):
        try:
            if self.pid is not None:
                process = psutil.Process(self.pid)
                self.memory_usage = process.memory_info().rss / (1024 * 1024)  # 转换为MB
                return self.memory_usage
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            self.memory_usage = 0
        return 0

def generate_data_wrapper(i, track_num, nw, py_dir, py_path, use_gs, process_info, extra_args=""):
    """包装原始的generate_data函数以获取PID和线程ID"""
    thread_id = threading.get_ident()
    logger.info(f"线程{i} 启动，线程ID: {thread_id}")
    
    n = track_num // nw
    command = f"{py_dir} {py_path} --data_idx {i*n} --data_set_size {n} --auto" + (" --use_gs" if use_gs else "") + (f" {extra_args}" if extra_args else "")
    logger.info(f"线程{i} 执行命令: {command}")
    
    # 使用subprocess.Popen创建子进程
    try:
        # 拆分命令为参数列表
        cmd_args = command.split()
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 存储子进程信息
        process_info.pid = process.pid
        process_info.subprocess = process
        
        logger.info(f"线程{i} 启动子进程，PID: {process.pid}")
        
        # 等待进程完成
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # 打印输出（可选）
        if stdout:
            logger.debug(f"线程{i} 标准输出: {stdout.decode('utf-8')}")
        if stderr:
            logger.debug(f"线程{i} 错误输出: {stderr.decode('utf-8')}")
            
        logger.info(f"线程{i} 完成，退出码: {return_code}")
        return thread_id, process.pid, return_code
    except Exception as e:
        logger.error(f"线程{i} 执行失败: {e}")
        return thread_id, None, 1

def memory_monitor(processes, memory_threshold=95):
    """
    监控系统内存使用情况，当总内存使用超过阈值时，杀掉占用内存最多的线程
    
    Args:
        processes: 正在运行的进程信息列表
        memory_threshold: 内存使用阈值百分比，默认95%
    """
    logger.info(f"内存监控启动，阈值设置为{memory_threshold}%")
    
    # 给进程一点时间启动并获取正确的PID
    time.sleep(3)
    
    # 记录已经终止的进程，避免重复终止
    terminated_processes = set()
    
    while any(not process.future.done() for process in processes):
        try:
            # 获取系统内存使用情况
            system_memory = psutil.virtual_memory()
            memory_percent = system_memory.percent
            
            # 更新每个进程的内存使用情况
            active_processes = [p for p in processes if not p.future.done()]
            for process in active_processes:
                process.update_memory_usage()
            
            # 过滤掉没有有效PID的进程
            active_processes_with_pid = [p for p in active_processes if p.pid is not None]
            
            if active_processes_with_pid:
                logger.debug(f"系统内存使用: {memory_percent:.1f}%, "
                          f"活动线程数: {len(active_processes_with_pid)}, "
                          f"内存使用情况: {[(p.thread_id, p.memory_usage) for p in active_processes_with_pid]}")
            
            # 如果内存使用超过阈值，杀掉占用最多内存的线程
            if memory_percent > memory_threshold and active_processes_with_pid:
                # 找出使用内存最多的进程，排除已经终止的进程
                available_processes = [p for p in active_processes_with_pid 
                                     if p.thread_id not in terminated_processes 
                                     and p.subprocess is not None]
                
                if available_processes:
                    max_memory_process = max(available_processes, key=lambda p: p.memory_usage)
                    
                    if max_memory_process.memory_usage > 0:
                        logger.warning(f"内存使用超过阈值: {memory_percent:.1f}% > {memory_threshold}%")
                        logger.warning(f"终止占用内存最多的线程 {max_memory_process.thread_id}，"
                                  f"内存使用: {max_memory_process.memory_usage:.1f}MB")
                        logger.warning(f"请减少nw参数，或检查任务是否最大时间过长")

                        try:
                            # 终止子进程而不是主进程
                            if max_memory_process.subprocess and max_memory_process.subprocess.poll() is None:
                                max_memory_process.subprocess.kill()
                                logger.info(f"已发送终止信号到子进程 PID: {max_memory_process.pid}")
                                
                            # 添加到已终止进程集合
                            terminated_processes.add(max_memory_process.thread_id)
                        except Exception as e:
                            logger.error(f"尝试终止进程时出错: {e}")
            
            # 每1秒检查一次
            time.sleep(1)
        except Exception as e:
            logger.error(f"内存监控发生错误: {e}")
            time.sleep(1)
    
    logger.info("所有进程已完成，内存监控退出")

if __name__ == "__main__":
    py_dir = os.popen('which python3').read().strip()

    parser = argparse.ArgumentParser(description='Run tasks with specified parameters. \ne.g. python3 os_run.py --robot_name airbot_play --task_name kiwi_place --track_num 100 --nw 8 --use_gs True')
    parser.add_argument('--robot_name', type=str, required=True, choices=["airbot_play", "mmk2","hand_arm"], help='Name of the robot')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task, see discoverse/examples/tasks_{robot_name}')
    parser.add_argument('--track_num', type=int, default=100, help='Number of tracks')
    parser.add_argument('--nw', type=int, required=True, default=8, help='Number of workers')
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    parser.add_argument('--memory_threshold', type=float, default=95.0, help='Memory threshold percentage (default: 95 percent)')
    parser.add_argument('--extra_args', type=str, default="", help='Extra arguments to pass to the task script')
    args = parser.parse_args()

    robot_name = args.robot_name
    task_name = args.task_name
    track_num = args.track_num
    nw = args.nw
    
    # Get number of logical processors
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    memory_per_worker = memory_gb / nw

    # 根据内存情况给出建议
    logger.info(f"系统信息: {cpu_count} 核心CPU, {memory_gb:.1f}GB 内存")
    logger.info(f"使用 {nw} 个工作进程，每进程最多可使用约 {memory_per_worker:.1f}GB 内存")

    # 改进的工作进程数建议
    if nw > cpu_count - 2 or memory_per_worker < 2.0:
        if memory_per_worker < 2.0:  # 每个工作进程少于2GB内存
            logger.warning(f"每个工作进程可用内存较少 ({memory_per_worker:.1f}GB)")
        if nw > cpu_count - 2:
            logger.warning(f"指定的工作进程数 ({nw}) 可能过多（系统逻辑处理器数量为 {cpu_count})")
        recommended_nw = min(int(memory_gb / 2), cpu_count - 2)
        logger.warning(f"建议的工作进程数: {recommended_nw}")
        logger.warning(f"继续使用用户指定的 {nw} 个工作进程...")
       
    use_gs = args.use_gs
    memory_threshold = args.memory_threshold
    
    py_path = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse/examples", f"tasks_{robot_name}/{task_name}.py")

    # 进程信息列表，用于跟踪每个线程的内存使用情况
    processes_info = []
    
    # 使用with语句创建线程池，它会在with块结束时自动关闭
    with ThreadPoolExecutor(max_workers=nw) as executor:
        # 提交所有任务并获取Future对象
        futures = []
        for i in range(nw):
            # 创建进程信息对象
            process_info = ProcessInfo(i, None)  # Future将在下面设置
            processes_info.append(process_info)
            
            # 提交任务，将process_info传递给任务
            future = executor.submit(generate_data_wrapper, i, track_num, nw, py_dir, py_path, use_gs, process_info, args.extra_args)
            process_info.future = future  # 设置Future
            futures.append(future)
        
        # 启动内存监控线程
        monitor_thread = threading.Thread(target=memory_monitor, 
                                         args=(processes_info, memory_threshold),
                                         daemon=True)
        monitor_thread.start()
        
        # 等待所有任务完成
        completed_count = 0
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    thread_id, pid, return_code = result
                    completed_count += 1
                    logger.info(f"工作进程 {i+1}/{nw} 完成 (线程ID: {thread_id}, PID: {pid}, 返回码: {return_code})")
                    logger.info(f"总进度: {completed_count}/{nw}")
            except Exception as e:
                logger.error(f"工作进程 {i+1} 执行失败: {e}")
        
        # 等待监控线程结束
        monitor_thread.join(timeout=1)

    logger.info("所有数据收集任务已完成！")
