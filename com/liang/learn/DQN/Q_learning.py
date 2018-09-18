# -*- coding: UTF-8 -*-

from com.liang.learn.utils.maze_env import Maze
from com.liang.learn.DQN.QLearningTable import QLearningTable


def update():
    # 学习100回合
    for episode in range(100):
        # 初始化state的观测值
        observation = env.reset()

        while True:
            # 更新可视化环境
            env.render()

            # RL大脑根据state的观测值挑选action
            action = RL.choose_action(str(observation))

            # 探索者在环境中实施这个action，并得到环境返回的下一个state观测值，reward和done（是否结束）
            observation_, reward, done = env.step(action)

            # RL从序列（state，action，reward，state_）中学习
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            # 游戏结束
            if done:
                break

    # 游戏结束
    print("game over")
    env.destroy()


if __name__ == '__main__':
    # 定义环境env和RL
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 开始可视化环境
    env.after(100, update)
    env.mainloop()
