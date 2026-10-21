# GraphQueryApp

Ứng dụng INET mô phỏng coordinator và worker cho bốn query class, ba physical plan và bảy chiến lược chọn plan. Mỗi query ghi canonical result, graph version, plan, payload/control/retry/switch bytes, CPU service và simulation time.

Input được materialize trong `output/current/cache/materialized`; output từng case nằm trong `output/current/simulation`.
