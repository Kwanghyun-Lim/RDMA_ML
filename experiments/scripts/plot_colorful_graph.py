import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

num_nodes = 16
file_objs = []
lines_of_file_objs = []
for i in range(num_nodes):
    if i == 0:
        file_name = 'server.log'
    else:
        file_name = 'worker' + str(i) + '.log'
    file_objs.append(open(file_name, 'r'))
    lines_of_file_objs.append(file_objs[-1].readlines())


fig, axes = plt.subplots(num_nodes, sharex=True)

WAIT = 0
COMPUTE = 1
PUSH = 2
RELAY = 3
is_relay = False

for i in range(num_nodes):
    lines_of_file_obj = lines_of_file_objs[i]
    for line in lines_of_file_obj:
        str_list = line.strip().split(' ')
        int_list = [int(str_list[0]), int(str_list[1]), int(str_list[2])]
        if i == 0:
            y_name = 'server'
        else:
            y_name = 'worker' + str(i)
            
        if int_list[2] == WAIT:
            pass
        elif int_list[2] == COMPUTE:
            axes[i].hlines(y_name, int_list[0], int_list[1], color ="red", linewidth=20)
        elif int_list[2] == PUSH:
            axes[i].hlines(y_name, int_list[0], int_list[1], color ="green", linewidth=20)
        elif int_list[2] == RELAY:
            is_relay = True
            axes[i].hlines(y_name, int_list[0], int_list[1], color ="blue", linewidth=20)
        
red_patch = mpatches.Patch(color='red', label='compute gradient in workers / update model in server')
green_patch = mpatches.Patch(color='green', label='push gradient in workers / push model in server')
blue_patch = mpatches.Patch(color='blue', label='relay model')

if is_relay:
    axes[0].legend(loc='lower left', handles=[red_patch, green_patch, blue_patch],
                   bbox_to_anchor=(0,1.02,1,0.2), mode='expand', ncol=2, fontsize='small')
else:
    axes[0].legend(loc='lower left', handles=[red_patch, green_patch],
                   bbox_to_anchor=(0,1.02,1,0.2), mode='expand', ncol=1, fontsize='small')
    
axes[num_nodes-1].set(xlabel='Time (ns)')
plt.savefig('colorful_graph.pdf')
