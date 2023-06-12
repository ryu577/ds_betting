import numpy as np


#Initialize betas
betas = np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]])


#Toy data X
x_5_rows = np.array([[1.0,1.0,1.0,1.0],
            [1.0,2.0,3.0,4.0],
             [1.0,1.0,1.0,1.0],
             [1.0,2.0,3.0,4.0],
             [1.0,1.0,1.0,1.0]])
#Toy data y
alpha_5_rows = np.array([[2.0,3.0],
                  [5.0,5.0],
                  [2.0,3.0],
                  [5.0,5.0],
                  [2.0,3.0]])



#matrix efficient gradient function
def compute_gradient_matrix_cookbook_5_rows(x_row,alpha_row,beta):
    return 2*(np.dot(beta,np.outer(x_row,x_row.T)).T - (np.outer(x_row,alpha_row)))

#step size for learning
step_size = .001


#gradient descent loop
for val in range(100):
    gradient_matrix_global = np.zeros(betas.shape)
    loss_overall = np.zeros((2,1))

    for row_idx in range(len(x_5_rows)):
        alpha_actual = np.expand_dims(alpha_5_rows[row_idx],-1)
        x_row = np.expand_dims(x_5_rows[row_idx],-1)
        alpha_hat = betas.dot(x_row)
        difference = alpha_actual-alpha_hat
        loss_overall+=difference

        gradient_matrix_local = compute_gradient_matrix_cookbook_5_rows(x_row,alpha_actual,betas).T
        gradient_matrix_global+= gradient_matrix_local
    betas += -step_size*gradient_matrix_global
    if val%10==0:
        print('epoch :',val,' loss:',loss_overall)