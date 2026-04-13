import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5,9), (9,13), (13, 14), (14, 15), (15, 16), (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)
]

df = pd.read_csv('final_dataset.csv')
word = 0
frames = df.iloc[word][1:2].iloc[0]
test_word = df.iloc[word][2:]
# print(frames)

test_word_list = test_word.to_list()

x = test_word_list[::3]
y = test_word_list[1::3]
z = test_word_list[2::3]


# vectors = []
# for i in range(0, frames):
#     start_idx = i * 63  # 21 landmarks * 3 coordinates
#     end_idx = start_idx + 63
#     frame_data = test_word_list[start_idx:end_idx]
#     frame = []
#     for i in range(0, len(frame_data), 3):
#         x = frame_data[i]
#         y = frame_data[i + 1]
#         z = frame_data[i + 2]
#         frame.append([x, y, z])
        
#     vectors.append(frame)

# with open('vectors.json', 'w') as f:
#     json.dump(vectors, f, indent=4)

# print(vectors)

# x, y, z = test_word_list

new = list(zip(x, y, z))
pd.DataFrame(new).to_csv('test_unity_landmarks_v2.csv', index=False, header=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.gca().invert_yaxis()
plt.gca().invert_zaxis()
plt.gca().invert_xaxis()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
for i, j in connections:
    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b')
plt.show()


