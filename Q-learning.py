import numpy
import pandas
import time
 
numpy.random.seed(2)
 
 
N_STATES = 10                    #������ �������
FIRST_STATE = 5                 #������ �������� ������
ACTIONS = ['left', 'right']
EPSILON = 0.7                     # ����� �������� ������� ��� ������� ��������
ALPHA = 0.3                         # �������� ����
GAMMA = 0.8                       # �������������� ������
MAX_EPISODES = 10             # ����� �������� (������ ��� ����� �������� �������� ��� ���������
FRESH_TIME = 0.05                # �������� �����������
FRESH_TIME_EPIS = 1
 
 
def build_q_table(n_states, actions):     # ���������� Q- �������
    table = pandas.DataFrame(numpy.zeros((n_states, len(actions))), columns=actions,)
    return table
 
 
def choose_action(state, q_table):             # ����� ��������, ���������� �������-greedy policy �� ���� � ������������ ������� ������� �������� � ������� ���������
    state_actions = q_table.iloc[state, :]
    if (numpy.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = numpy.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name
 
 
def get_env_feedback(S, A):  # �������
    if A == 'right':
        if S == N_STATES-1:
            S_ = 'terminal'
            R = 1    # ������� � ���������
        else:
            S_ = S + 1
            R = 0
    else:
       # R = 0
        if S == 0:
            S_ = 'abyss'
            R = -1   # ������� � ��������
 
        else:
            S_ = S - 1
            R = 0
    return S_, R
 
 
def update_enviroment(S, episode, step_counter):   # ����������� "�������"
    env_list = ['abyss:']+['-']*(N_STATES-2) + [':treasure']
    if S == 'terminal' or S == 'abyss':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME_EPIS)
        print('\r                                ', end='')
 
    else:
        env_list[S] = '(o.0)'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
 
 
def reinforsment_learning():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = FIRST_STATE
        is_terminated = False
        update_enviroment(S, episode, step_counter)
        while is_terminated == False:
            A = choose_action(S, q_table)                    # ������� �������� � ��������������� ������
            S_, R = get_env_feedback(S, A)                   #  �������� ����� ����� �� ��� �������� (��������� �������� � �������)
            q_predict = q_table.ix[S, A]                     #  ������������ �������� �������� �������� �� ������������ �������
            if S_ != 'terminal' and S_ != 'abyss':
 
                # ��������� ����� �������:
                if A == 'right': # ��� ���� ����� ���� ���������� � ������ ���� ��� ������ - ���������, ����� - ��������
                    q_target = R + GAMMA * q_table.iloc[S_, :].max()  # ���������� �������� �������� ����� ������� + ������������ �������� �������� �������� � ��������� ������!!!!!????
                else:
                    q_target = R + GAMMA * q_table.iloc[S_, :].min()
 
            else:
                q_target = R
                is_terminated = True
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)
            S = S_
 
 
            update_enviroment(S, episode, step_counter+1)
            step_counter += 1
        
    return q_table
 
 
q_table = reinforsment_learning()
print('\r\nQ-table:\n')
print(q_table)