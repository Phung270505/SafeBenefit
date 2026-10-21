# OMNeT++/INET simulator

`graph_query/` chứa GraphQueryApp, NED và cấu hình INET. `tools/build.sh` build shared library; `tools/run_case.sh` chạy một dataset/split/method.

Người dùng không cần gọi hai shell helper trực tiếp. [run.py](../../run.py) điều khiển build và toàn bộ case qua WSL `OMNeT-opp_env`.

Simulator hỗ trợ ba plan và bảy method ID đăng ký trong `config/study.json`. Wire bytes được đọc từ scalar `txPk:sum(packetBytes)` tại MAC; payload, control, retry và switch bytes được ghi riêng.
