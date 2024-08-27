from Keyrate_Evaluation import *
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

def plot_network_noise_vs_channel_noise(full_size, honest_sizes):
    # define q values
    q_values = np.linspace(0, 1, 100)

    for honest_size in honest_sizes:
        # initialize lists to store Qx and pstar values
        Qx_values = []
        pstar_values = []

        # loop through q values and calculate Qx and pstar for each
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # append Qx and pstar values to lists
            Qx_values.append(network_obj.Qx)
            pstar_values.append(network_obj.pstar)

        # plot Qx and pstar as a function of channel noise
        plt.plot(q_values, pstar_values, label=f'Honest repeaters: {honest_size}')
    plt.plot(q_values, Qx_values, linestyle='dotted', color='black')
    plt.text(0.4, 0.5, '$w(Q_x)$', color='black', fontsize=12, ha='center', va='center')
    plt.grid(True)
    plt.ylim(0, .6)
    plt.xlim(0, .5)
    plt.xlabel('Depolarizing Channel Parameter $q$')
    plt.ylabel('$p^*$')
    #plt.title(f'{full_size} total links, {honest_size} honest links')
    plt.legend()
    plt.show()

def plot_keyrate_vs_signalrounds(q, full_size, honest_sizes):
    # define range of signal rounds
    signal_rounds = np.logspace(4, 15, num=10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        # network object
        network_obj = network(full_size, honest_size, depolarization(q))
        # initialize list to store key rates
        keyrates = []
        # loop through signal rounds and calculate key rate for each
        for signal in signal_rounds:
            keyrate_val = keyrate(signal, network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_val)
        # plot key rate as a function of signal rounds
        plt.plot(signal_rounds, keyrates, label=f'Honest repeaters: {honest_size}')
    #BB84
    BB84_keyrates = []
    #fully corrupt network
    network_obj = network(full_size, 0, depolarization(q))
    for signal in signal_rounds:
        keyrate_val = BB84_F(signal, network_obj.Qx)
        BB84_keyrates.append(keyrate_val)
    plt.plot(signal_rounds, BB84_keyrates, color='black', label='BB84-F', linestyle='dotted', linewidth=2)

    #plt.grid(True)
    plt.xscale('log')
    plt.yscale('linear')
    plt.ylim(0, .3)
    plt.xlim(1e5, 1e11)
    plt.xlabel('Number of Signals $N$')
    plt.ylabel('Finite Key-Rate')
    #plt.title(f'{full_size} total links, {q*100}% channel depolarization')
    plt.legend(loc='upper left')
    plt.show()
    

def plot_keyrate_vs_Qx(full_size, honest_sizes, N, inset=False):
    # define q values
    q_values = np.linspace(0, 1, 10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        #initialize lists to store Qx values
        Qx_values = []
        # initialize list to store key rates
        keyrates = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # append Qx values to list
            Qx_values.append(network_obj.Qx)
            keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_val)
        # plot key rate as a function of Qx
        plt.plot(Qx_values, keyrates, label=f'Honest repeaters: {honest_size}')
    plt.plot(np.linspace(0,.5,1000), BB84_F(N, np.linspace(0,.5,1000)), label='BB84-F', color='black', linestyle='dotted', linewidth=2)
    #plt.grid(True)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlim(0,.15)
    plt.ylim(1e-4, 1)
    plt.xlabel('Noise $w(Q_x)$')
    plt.ylabel('Finite Key-Rate')
    #plt.title(f'Finite-Key Rates for {full_size} Total Links, {N:.1e} Signal Rounds')
    plt.legend()

    if inset==True:
        # Create inset of width 30% and height 30% of the parent axes' bounding box at the lower left corner (at 0.05, 0.1)
        axins = plt.gca().inset_axes([.05, .1, 0.3, 0.3])
        for honest_size in honest_sizes:
            Qx_values = []
            keyrates = []
            for q in q_values:
                network_obj = network(full_size, honest_size, depolarization(q))
                Qx_values.append(network_obj.Qx)
                keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
                keyrates.append(keyrate_val)
            axins.plot(Qx_values, keyrates, label=f'Honest links: {honest_size}')
        axins.plot(np.linspace(0,.5,1000), BB84_F(N, np.linspace(0,.5,1000)), label='BB84-F', color='black', linestyle='dotted', linewidth=2)
        axins.set_xlim(0, .05)  # apply the x-limits
        axins.set_ylim(.2, 1)  # apply the y-limits
        axins.set_xscale('linear')
        axins.set_yscale('log')
        axins.set_xticklabels([])  # remove xtick labels
        axins.set_yticks([])  # remove yticks
        axins.set_yticklabels([], minor=True)  # remove ytick labels

        # Add rectangle and connecting lines from the rectangle to the inset axes
        plt.gca().indicate_inset_zoom(axins, edgecolor="black")

    plt.show()

def plot_asymptotic_keyrate_vs_Qx(full_size, honest_sizes):
    # define q values
    q_values = np.linspace(0, 1, 10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        #initialize lists to store Qx values
        Qx_values = []
        # initialize list to store key rates
        keyrates = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # append Qx values to list
            Qx_values.append(network_obj.Qx)
            keyrate_inf_val = keyrate_inf(network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_inf_val)
        # plot key rate as a function of Qx
        plt.plot(Qx_values, keyrates, label=f'Honest repeaters: {honest_size}')
    
    plt.plot(np.linspace(0,.5,1000), BB84_A(np.linspace(0,.5,1000)), color = 'black', label='BB84-A', linestyle='dotted', linewidth=2)
    #plt.grid(True)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlim(0,.2)
    plt.ylim(1e-4, 1)
    plt.xlabel('Noise $w(Q_x)$')
    plt.ylabel('Asymptotic Key-Rate')
    #plt.title(f'Asymptotic Key Rates for {full_size} Total Links')
    plt.legend()
    plt.show()

def plot_noise_tolerance_vs_honest_links(full_size, honest_sizes, N):
    # define q values
    q_values = x = np.linspace(0, 1, 1000)
    # initialize lists to store noise tolerances
    tolerances = []
    inf_tolerances = []
    #honest ratio list

    honest_ratios = []
    # loop through honest sizes
    for honest_size in honest_sizes:
        # initialize lists
        honest_ratios.append(honest_size/full_size)
        keyrates = []
        inf_keyrates = []
        Qx_values = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # calculate key rate
            keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
            inf_keyrate_val = keyrate_inf(network_obj.Qx, network_obj.pstar)
            Qx_values.append(network_obj.Qx)
            # append key rate to list
            keyrates.append(max(0,keyrate_val))
            inf_keyrates.append(max(0,inf_keyrate_val))
        # calculate noise tolerance
        tolerance = noise_tolerance(keyrates, Qx_values)
        inf_tolerance = noise_tolerance(inf_keyrates, Qx_values)
        # append noise tolerance to list
        tolerances.append(tolerance)
        inf_tolerances.append(inf_tolerance)

    bb84_keyrates = []
    for noise in Qx_values:
        bb84_keyrates.append(max(0,BB84_F(N, noise)))
    finite_BB84_tolerance = noise_tolerance(bb84_keyrates, Qx_values)
    
    #splines
    finite_spline = PchipInterpolator(honest_ratios, tolerances)
    inf_spline = PchipInterpolator(honest_ratios, inf_tolerances)
    
    # plot noise tolerance as a function of honest links
    plt.plot(x, inf_spline(x), label='Asymptotic')
    plt.plot(x, finite_spline(x), label='Finite-Key')
    
    plt.axhline(y=0.11, color='black', linestyle='--', label='BB84-A')
    plt.axhline(y=finite_BB84_tolerance, color='black', linestyle='dotted', label='BB84-F')
    
    plt.plot(honest_ratios, tolerances, 'o')
    plt.plot(honest_ratios, inf_tolerances, 'o')
    
    plt.xlabel('Ratio of Honest Repeaters')
    plt.ylabel('Noise Tolerance')
    plt.legend()
    plt.show()

#==========================General Network Keyrate Plots==========================

def plot_general_keyrate_vs_signalrounds(Qx_array, p, m):
    # define range of signal rounds
    signal_rounds = np.logspace(4, 12, num=1000)
    for Qx in Qx_array:
        keyrates = []
        # loop through signal rounds and calculate key rate for each
        for signal in signal_rounds:
            keyrate_val = general_network_keyrate(Qx, signal, p, m)
            keyrates.append(keyrate_val)
        # plot key rate as a function of signal rounds
        plt.plot(signal_rounds, keyrates, label=f'$w(Q_x)$ = {Qx}')
        #plt.plot(signal_rounds, keyrates, label=f'k = {len(p)}')

    plt.gca().set_prop_cycle(None)
    for Qx in Qx_array:
        #BB84
        BB84_keyrates = []
        #fully corrupt network
        for signal in signal_rounds:
            keyrate_val = BB84_F(signal, Qx)
            BB84_keyrates.append(keyrate_val)
        plt.plot(signal_rounds, BB84_keyrates, label=f'BB84-F ({int(Qx*100)}% Noise)', linestyle='dotted', linewidth=2)

    #plt.grid(True)
    plt.xscale('log')
    plt.yscale('linear')
    plt.ylim(bottom = 0)
    plt.xlim(1e5, 1e11)
    plt.xlabel('Number of Signals $N$')
    plt.ylabel('General Network Finite Key-Rate')
    #plt.title(f'{full_size} total links, {q*100}% channel depolarization')
    plt.legend(loc='upper left')
    #plt.show()