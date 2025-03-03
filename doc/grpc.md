## grpc

仿真器提供了grpc的例程，并与airbot系列机器人产品使用相同的接口。

### 安装

```bash
pip install grpcio
pip install grpcio-tools
cd discoverse/examples/grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. array.proto
```

### 使用

```bash
(terminal 1)
python3 airbot_play_grpc_server.py

(terminal 2)
python3 airbot_play_grpc_client.py
```
