import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)



df = pd.read_csv('real_data_obvious_decision.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])


def compute_gradient_matrix_cookbook_5_rows(x_row,alpha_row,beta):
    return 2*(np.dot(beta,np.outer(x_row,x_row.T)).T - (np.outer(x_row,alpha_row)))

def alpha_function(val):
    if val>=3:
        return 50
    else:
        return 10

df['alpha'] = df['bb'].apply(alpha_function)
df['beta'] = 60-df['alpha']


step_size = .0001
epochs = 50
features = ['constant','opponent_lineup_bb_pct']

a1_real = df['alpha'].values
b2_real = df['beta'].values


alpha_real = np.stack([a1_real,b2_real],axis=1)
x_real = df[features].to_numpy()
betas_real = np.ones((2,x_real.shape[1]))

for val in range(epochs):
    gradient_matrix_global = np.zeros(betas_real.shape)
    loss_overall = np.zeros((2,1))

    for row_idx in range(len(x_real)):
        alpha_actual = np.expand_dims(alpha_real[row_idx],-1)
        x_row = np.expand_dims(x_real[row_idx],-1)
        alpha_hat = betas_real.dot(x_row)
        difference = alpha_actual-alpha_hat
        loss_overall+=difference

        gradient_matrix_local = compute_gradient_matrix_cookbook_5_rows(x_row,alpha_actual,betas_real).T
        gradient_matrix_global+= gradient_matrix_local
    betas_real += -step_size*gradient_matrix_global
    print(loss_overall)

walks = df['bb'].values

betas_correct = np.array([[-250/3,4000/3],[700/3,-4000/3]])

for row_idx in range(50):
    alpha_actual = np.expand_dims(alpha_real[row_idx],-1)
    x_row = np.expand_dims(x_real[row_idx],-1)
    alpha_hat = betas_correct.dot(x_row)
    print(alpha_hat,alpha_real[row_idx],walks[row_idx])
    print()