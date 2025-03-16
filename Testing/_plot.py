import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import savgol_filter

Recovered_loads = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/__Thesis Report/restored_loads.npy")
Recovered_loads *= 100
times = np.arange(1, 25)

plt.figure(figsize=(8, 6))

plt.plot(times, Recovered_loads, color='purple', linewidth=3)

plt.xlabel('Time (h)')
plt.ylabel('Recovered Load (%)')

plt.xlim(0, 25)
plt.ylim(-1, 105)

for y in range(0, 102, 20):
    plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

plt.grid(False)
plt.xticks(range(0, 25, 2))  # X-axis ticks every 2 hours
plt.yticks(range(0, 105, 20))

plt.tight_layout()

plt.show()

# RC_resources = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/__Thesis Report/RCs_resources.npy")
# times = np.arange(1, 25)

# plt.figure(figsize=(8, 6))

# plt.plot(times, RC_resources[1], 'g-', marker='o', markersize=6, linewidth=1)

# plt.xlabel('Time (h)')
# plt.ylabel('Resource (unit)')

# plt.xlim(0, 24)
# plt.ylim(-1, 60)

# for y in range(0, 58, 7):
#     plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

# plt.grid(False)
# plt.xticks(range(0, 25, 2))  # X-axis ticks every 2 hours
# plt.yticks(range(0, 63, 7))

# # ax = plt.gca()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)

# plt.tight_layout()

# plt.show()

# times = np.arange(1, 25)  # Time points from 1 to 24 hours
# n_points = len(times)

# CS_0_1 = np.zeros(n_points)
# CS_0_1[18:] = 1
# CS_0_2 = np.zeros(n_points)
# CS_0_2[17:] = 1
# CS_1_1 = np.zeros(n_points)
# CS_1_1[7:] = 1
# CS_1_2 = np.zeros(n_points)
# CS_1_2[16:] = 1
# CS_2_1 = np.zeros(n_points)
# CS_2_1[8:] = 1
# CS_2_2 = np.zeros(n_points)
# CS_2_2[9:] = 1

# plt.figure(figsize=(10, 6))
# plt.bar(times, CS_1_1, color='#8B0000', label='CS_1_1', width=0.5)
# plt.bar(times, CS_2_1, bottom=CS_1_1, color='#006400', label='CS_2_1', width=0.5)
# plt.bar(times, CS_2_2, bottom=CS_1_1+CS_2_1, color='#00008B', label='CS_2_2', width=0.5)
# plt.bar(times, CS_1_2, bottom=CS_1_1+CS_2_1+CS_2_2, color='#FF6666', label='CS_1_2', width=0.5)
# plt.bar(times, CS_0_2, bottom=CS_1_1+CS_2_1+CS_2_2+CS_1_2, color='#90EE90', label='CS_0_2', width=0.5)
# plt.bar(times, CS_0_1, bottom=CS_1_1+CS_2_1+CS_2_2+CS_1_2+CS_0_2, color='#ADD8E6', label='CS_0_1', width=0.5)

# plt.legend()

# plt.xlabel('Time (h)')
# plt.ylabel('Damaged EV chargers')
# plt.ylim(0, 6)
# plt.xlim(0, 24)

# plt.xticks(ticks=np.arange(0, 25, 2), labels=np.arange(0, 25, 2))

# ax = plt.gca()
# ax2 = ax.twinx()
# ax2.set_ylim(0, 6)
# ax2.set_ylabel('Repair Crew')
# ax2.set_yticks([1, 2, 3, 4, 5, 6])
# ax2.set_yticklabels(['RC1', 'RC2', 'RC3', 'RC1', 'RC2', 'RC3'])

# plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
# plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
# plt.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
# plt.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
# plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
# plt.axhline(y=6, color='gray', linestyle='--', alpha=0.5)

# plt.legend()
# plt.tight_layout()

# plt.show()

# ppo_model = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results/mappo/trial_5/episode_rewards_3000286.npy")
# # smooth = savgol_filter(ppo_model, window_length=11, polyorder=2)
# ippo_model = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results/ippo/trial_5/episode_rewards_3000191.npy")
# trpo_model = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results/matrpo/trial_3/episode_rewards_3000222.npy")
# a2c_model = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results/maa2c/trial_1/episode_rewards_3000133.npy")
# ddpg_model = np.load("D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results/maddpg/trial_2/episode_rewards_3000540.npy")

# episodes = range(len(ppo_model))

# plt.figure(figsize=(12, 8), dpi=100)

# plt.plot(episodes, ppo_model, label='MAPPO', color='b', linestyle='-', linewidth=2)
# plt.plot(episodes, ippo_model, label='IPPO', color='r', linestyle='-', linewidth=2)
# # plt.plot(episodes, smooth, label='smooth', color='r', linestyle='-', linewidth=2)
# # plt.plot(episodes, trpo_model, label='MATRPO', color='g', linestyle='--', linewidth=2)
# # plt.plot(episodes, a2c_model, label='MAA2C', color='r', linestyle='-.', linewidth=2)
# # plt.plot(episodes, ddpg_model, label='MADDPG', color='orange', linestyle=':', linewidth=2)

# # plt.title("Evolution of Episodic Reward During Training", fontsize=14, weight='bold')
# plt.xlabel("1e4 timesteps", fontsize=12)
# plt.ylabel("Reward", fontsize=12)

# plt.legend(title="MARL approaches", loc='upper left', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()

# plt.show()


# print(len(ppo_model))
# print(len(trpo_model))
# print(len(a2c_model))
# print(len(ddpg_model))

