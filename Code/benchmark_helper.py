from matplotlib import rc
import matplotlib.pyplot as plt

import numpy as np
from random import randint, uniform
from sklearn.metrics import accuracy_score

from attacks import *

normalization = 255
width = 28
heigth = 28

def normalize(X):
    return X / normalization

def denormalize(X):
    return X * normalization

def show_samples(X, nrows=5, ncols=5, rsize=5, csize=5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row', 
                           gridspec_kw={'hspace': -0.02, 'wspace': -0.02}, figsize=(rsize, csize))

    counter=0
    for row in ax:
        for col in row:
            col.imshow(X[counter], 'gray')
            col.axis('off')
            counter += 1
    plt.show()

def benchmark_attack_by_model(X, y_labels, model=None, attack_model=None, samples=100, num_steps=15):
    X_norm = normalize(X)
    iter_res_dict = {}

    norms_range = np.linspace(0, 0.5, num=num_steps)

    for i in range(samples):
        x_normalized = np.expand_dims(X_norm[i, :], axis=0)
        y = np.zeros(len(y_labels))
        true_class_pos = model.predict(x_normalized)
        y[true_class_pos] = 1
    
        for norm in norms_range:
            attack = attack_model()
            if type(attack).__name__ in ['AttackNoise']:
                attack.attack(x_normalized, y, model.predict, max_norm=norm)
            elif type(attack).__name__ in ['AttackIFGSM', 'AttackTIFGSM', 'AttackDeepFool', 'AttackFGSM']:
                attack.attack(x_normalized, model.W, model.b, y, model.predict, max_norm=norm)
            
            elif type(attack).__name__ in ['AttackMIFGSM']:
                attack.attack(x_normalized, model.W, model.b, y, model.predict, max_norm=norm, momentum=0.95)
        
            success = attack.tricked
            
            if norm in iter_res_dict.keys():
                iter_res_dict[norm]["appearance_count"] += 1
                iter_res_dict[norm]["success_count"] += success
            else:
                iter_res_dict.update({norm: {"appearance_count": 1, "success_count": success} })

    success_rate = {}
    for key, value in iter_res_dict.items():
        success_rate[key] = value["success_count"] / value["appearance_count"] * 100

    perturbations_list = [x[0] for x in sorted(success_rate.items(), key=lambda x: x[0]) ]
    successrate_list = [x[1] for x in sorted(success_rate.items(), key=lambda x: x[0]) ]

    return perturbations_list, successrate_list, type(attack).__name__

def show_adversarial_sample(x, y_labels, true_class=[], model=None, attack_model=None, max_norm=10, max_iters=10):

    x_normalized = normalize(x)
    attack = attack_model()

    if type(attack).__name__ in ['AttackNoise']:
        x_adv_normalized = attack.attack(x_normalized, true_class, model.predict, max_norm=max_norm)
    
    elif type(attack).__name__ in ['AttackIFGSM', 'AttackTIFGSM', 'AttackFGSM', 'AttackDeepFool', 'AttackMIFGSM']:
        x_adv_normalized = attack.attack(x_normalized, model.W, model.b, true_class, model.predict, max_norm=max_norm)
    
    elif type(attack).__name__ in ['AttackMIFGSM']:
        x_adv_normalized = attack.attack(x_normalized, model.W, model.b, true_class, model.predict, max_norm=max_norm, momentum=0.1)
        
    x_adv = denormalize(x_adv_normalized)
    x_adv_pred = model.predict(np.array(x_adv_normalized))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, squeeze=True, sharex=True, sharey=True)
    fig.set_size_inches(15,5)
    fig.suptitle('Adversarial example')
    
    ax1.imshow(x.reshape(28,28), 'gray')
    ax1.set_title(f"Original\n prediction: {model.predict(x_normalized)[0]}")
    
    ax2.imshow(x.reshape(28,28) - x_adv.reshape(28,28), "gray")
    ax2.set_title(f"Difference")
    
    ax3.imshow(x_adv.reshape(28,28), 'gray')
    ax3.set_title(f"Adversarial\n prediction: {x_adv_pred[0]}")
    plt.show()

    pred_with_labels = model.predict_by_labels(np.array(x_adv_normalized), y_labels)
    plt.bar(pred_with_labels[0].keys(), pred_with_labels[0].values())
    plt.show()
    
    print(f"Norm_inf: {LA.norm(abs(x_normalized - x_adv_normalized), np.inf)}; Result: {attack.tricked}")
    print(f"Max value: {max(max(x_adv))}")
    print(f"Min value: {min(min(x_adv))}")
    
def get_adv_by_model_norm(X, Y_true, model=None, attack=None, norm=0.1):
    X_adv = None
    
    atck = attack()
    if type(atck).__name__ in ['AttackNoise']:
        X_adv = [atck.attack(X[i, :].reshape(1,-1), Y_true[i], model.predict, max_norm=norm)
                 for i in range(X.shape[0])]
    
    elif type(atck).__name__ in ['AttackIFGSM', 'AttackTIFGSM', 'AttackDeepFool', 'AttackFGSM']:
        X_adv = [atck.attack(X[i, :].reshape(1,-1), model.W, model.b, Y_true[i, :], model.predict, max_norm=norm)
                 for i in range(X.shape[0])]
    
    elif type(atck).__name__ in ['AttackMIFGSM']:
        X_adv = [atck.attack(X[i, :].reshape(1,-1), model.W, model.b, Y_true[i], model.predict, max_norm=norm, momentum=0.95)
                 for i in range(X.shape[0])]

    return np.array(X_adv).reshape(X.shape[0], -1)

def benchmark_defense_by_model(X, Y_true, y_true, model=None, attack=None, samples=100, num_steps=0, a=0, b=0.5):
    norms = np.linspace(a, b, num=num_steps)
    scores = np.zeros(norms.shape)

    for idx, norm in enumerate(norms):
        X_normalize = normalize(X[0:samples])
        y_true = y_true[0:samples].reshape(samples)

        X_adv = get_adv_by_model_norm(X_normalize, Y_true, model=model, attack=attack, norm=norm)
        scores[idx] = accuracy_score(model.predict(X_adv), y_true)
    return norms, scores, attack.__name__

def benchmark_attack_by_model(X, Y_true, y_true, model=None, attack=None, samples=100, num_steps=0, a=0, b=0.5):
    y_correct_idx = ((model.predict(X) == y_true) == y_true)

    norms = np.linspace(a, b, num=num_steps)
    scores = np.zeros(norms.shape)

    for idx, norm in enumerate(norms):
        X_normalize = normalize(X[y_correct_idx][0:samples])
        y_true = y_true[0:samples].reshape(samples)

        X_adv = get_adv_by_model_norm(X_normalize, Y_true[y_correct_idx], model=model, attack=attack, norm=norm)
        scores[idx] = 1 - accuracy_score(model.predict(X_adv), y_true)
    return norms, scores, attack.__name__


def benchmark_defense_models_visual(X, Y_true, y_true, model=None, samples=100, num_steps=10):
    x1, y1, label1 = benchmark_defense_by_model(X, Y_true, y_true, model=model, attack=AttackFGSM, samples=samples, num_steps=num_steps)
    x2, y2, label2 = benchmark_defense_by_model(X, Y_true, y_true, model=model, attack=AttackIFGSM, samples=samples, num_steps=num_steps)
    x3, y3, label3 = benchmark_defense_by_model(X, Y_true, y_true, model=model, attack=AttackMIFGSM, samples=samples, num_steps=num_steps)
    x4, y4, label4 = benchmark_defense_by_model(
        X, Y_true, y_true, model=model, attack=AttackDeepFool, samples=samples, num_steps=num_steps
    )

    fig, ax = plt.subplots(num=None, figsize=(12, 6), dpi=100, facecolor='w')
    plt.grid(which='major', linestyle='-', color='black', alpha=0.2)
    ax.set_title('Perturbation vs Accuracy Rate')
    ax.set_xlabel(r"Perturbation $\epsilon$")
    ax.set_ylabel('Accuracy (%)')

    plt.plot(np.array(x1), y1, linestyle='--', marker='o', markersize=2, color='#00A658', label = label1)
    plt.plot(np.array(x2), y2, linestyle='--', marker='o', markersize=2, color='#2300A8', label = label2)
    plt.plot(np.array(x3), y3, linestyle='--', marker='o', markersize=2, color='red', label = label3)
    plt.plot(np.array(x4), y4, linestyle='--', marker='o', markersize=2, color='orange', label = label4)

    plt.legend()

def benchmark_attack_models_visual(X, y_labels, model=None, samples=100, num_steps=10):
    x1, y1, label1 = benchmark_attack_by_model(X, y_labels, model=model, attack_model=AttackFGSM, samples=samples, num_steps=num_steps)
    x2, y2, label2 = benchmark_attack_by_model(X, y_labels, model=model, attack_model=AttackIFGSM, samples=samples, num_steps=num_steps)
    x3, y3, label3 = benchmark_attack_by_model(X, y_labels, model=model, attack_model=AttackMIFGSM, samples=samples, num_steps=num_steps)
    x4, y4, label4 = benchmark_attack_by_model(
        X, y_labels, model=model, attack_model=AttackDeepFool, samples=samples, num_steps=num_steps
    )

    fig, ax = plt.subplots(num=None, figsize=(12, 6), dpi=100, facecolor='w')
    plt.grid(which='major', linestyle='-', color='black', alpha=0.2)
    ax.set_title('Perturbation vs Success Rate')
    ax.set_xlabel(r"Perturbation $\epsilon$")
    ax.set_ylabel('Success  (%)')

    plt.plot(np.array(x1), y1, linestyle='--', marker='o', markersize=2, color='#00A658', label = label1)
    plt.plot(np.array(x2), y2, linestyle='--', marker='o', markersize=2, color='#2300A8', label = label2)
    plt.plot(np.array(x3), y3, linestyle='--', marker='o', markersize=2, color='red', label = label3)
    plt.plot(np.array(x4), y4, linestyle='--', marker='o', markersize=2, color='orange', label = label4)

    plt.legend()

def compare_model_for_two_inputs(X1, X2, Y_true, y_true, model=None, attack=None, samples=100, num_steps=10, visualize=True):
    x1, y1, label1 = benchmark_defense_by_model(X1, Y_true, y_true, model=model, attack=attack, samples=samples, num_steps=num_steps)
    x2, y2, label2 = benchmark_defense_by_model(X2, Y_true, y_true, model=model, attack=attack, samples=samples, num_steps=num_steps)
    
    if visualize:
        fig, ax = plt.subplots(num=None, figsize=(12, 6), dpi=100, facecolor='w')
        plt.grid(which='major', linestyle='-', color='black', alpha=0.2)
        ax.set_title('Perturbation vs Accuracy Rate')
        ax.set_xlabel(r"Perturbation $\epsilon$")
        ax.set_ylabel('Accuracy (%)')

        plt.plot(x1, y1, linestyle='--', marker='o', markersize=2, color='#00A658', label = f"clean_{label1}")
        plt.plot(x2, y2, linestyle='--', marker='o', markersize=2, color='#2300A8', label = f"defended_{label2}")
        plt.legend()
        
    a = round(sum(y1/num_steps), 4)
    b = round(sum(y2/num_steps), 4)
    return a,b
