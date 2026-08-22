# 第 12 章代码：warp、CTA、mbarrier 与 cluster 同步

本目录提供三个相互独立的同步协议：

- `warp_collectives.py`：warp barrier、elect、vote、shuffle、redux，以及 named CTA barrier/reduction；
- `mbarrier_pingpong.py`：双 warp、双 stage 的 full/empty mbarrier 环形复用；
- `cluster_dsmem_ring.py`：SM90+ CTA cluster、DSMEM、remote mbarrier 与 `mapa`。

在仓库根目录运行：

```bash
python discussion/code/12_synchronization/warp_collectives.py
python discussion/code/12_synchronization/mbarrier_pingpong.py
python discussion/code/12_synchronization/cluster_dsmem_ring.py --cluster-size 2
```

所有程序都进行数值/协议结果验证，并检查保留 PTX 中的关键指令族。不要通过故意少 arrive 或错 phase 的方式在 GPU 上测试死锁；这类错误应通过参与者表、状态机推演和有时限的等待诊断。
