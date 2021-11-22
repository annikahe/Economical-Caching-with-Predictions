import pickle
import matplotlib.pyplot as plt

with open('Tests/Instances/ex_talk.pkl', 'rb') as inp:
    mindet = pickle.load(inp)
    print(mindet.mindet_history.purchases)

    opt_off = pickle.load(inp)
    print(opt_off.purchases)

    mindet.plot_history_mindet_sections()
    plt.show()
