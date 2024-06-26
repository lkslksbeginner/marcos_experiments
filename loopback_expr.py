#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import external  # imports external.py

import experiment as ex
from local_config import grad_board

import pdb
st = pdb.set_trace

# In each TR, a wait of rf_start_delay occurs, the RF turns on, it
# turns off after rf_length. Each TR is by default 1 second long. The RX
# turns on rx_pad microseconds before and turns off rx_pad
# microseconds after the RF pulse.

lo_freq = 5 # MHz
rx_period = 5 # us
rx_pad = 20 # us
rf_start_delay = 100 # us
rf_amp = 0.4
rf_length = 200 # us

def rxplot_src(rxd_iq):
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    if 'rx0' in rxd_iq:
        rx0_shape = np.shape(rxd_iq['rx0'])
        print(f'Shape of rx0 data: {rx0_shape}')
        rx0_data = rxd_iq['rx0']
        rx0_i = np.real(rx0_data)
        rx0_q = np.imag(rx0_data)
    # Plot rx0_i
        axs[0, 0].plot(rx0_i, label='rx0_i')
        axs[0, 0].set_title('rx0_i')
        axs[0, 0].legend()

        # Plot rx0_q
        axs[0, 1].plot(rx0_q, label='rx0_q')
        axs[0, 1].set_title('rx0_q')
        axs[0, 1].legend()
    if 'rx1' in rxd_iq:
        rx1_shape = np.shape(rxd_iq['rx1'])
        print(f'Shape of rx1 data: {rx1_shape}')
        rx1_data = rxd_iq['rx1']
        rx1_i = np.real(rx1_data)
        rx1_q = np.imag(rx1_data)
        # Plot rx1_i
        axs[1, 0].plot(rx1_i, label='rx1_i')
        axs[1, 0].set_title('rx1_i')
        axs[1, 0].legend()

        # Plot rx1_q
        axs[1, 1].plot(rx1_q, label='rx1_q')
        axs[1, 1].set_title('rx1_q')
        axs[1, 1].legend()

def rxplot(rxd_iq):
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    if 'rx0' in rxd_iq:
        rx0_data = rxd_iq['rx0']
        rx0_i = np.real(rx0_data)
        rx0_q = np.imag(rx0_data)

        # Subplot 1: rx0_i and rx0_q
        axs[0, 0].plot(rx0_i, label='rx0_i')
        axs[0, 0].plot(rx0_q, label='rx0_q')
        axs[0, 0].set_title('rx0_i and rx0_q')
        axs[0, 0].legend()

        # Subplot 2: Amplitude and Argument of rx0_i and rx0_q
        axs[0, 1].plot(np.abs(rx0_data), label='Amplitude')
        axs[0, 1].plot(np.angle(rx0_data), label='Argument')
        axs[0, 1].set_title('Amplitude and Argument of rx0')
        axs[0, 1].legend()

    if 'rx1' in rxd_iq:
        rx1_data = rxd_iq['rx1']
        rx1_i = np.real(rx1_data)
        rx1_q = np.imag(rx1_data)

        # Subplot 3: rx1_i and rx1_q
        axs[1, 0].plot(rx1_i, label='rx1_i')
        axs[1, 0].plot(rx1_q, label='rx1_q')
        axs[1, 0].set_title('rx1_i and rx1_q')
        axs[1, 0].legend()

        # Subplot 4: Amplitude and Argument of rx1_i and rx1_q
        axs[1, 1].plot(np.abs(rx1_data), label='Amplitude')
        axs[1, 1].plot(np.angle(rx1_data), label='Argument')
        axs[1, 1].set_title('Amplitude and Argument of rx1')
        axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
    



    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def long_loopback(rf_interval=1000000, trs=20):

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, halt_and_reset=True)

    for k in range(trs):
        rf_t = rf_start_delay + k*rf_interval + np.array([0, rf_length])
        rx_t = rf_start_delay + k*rf_interval - rx_pad + np.array([0, rf_length + 2*rx_pad])
        expt.add_flodict({
            'tx0': ( rf_t, np.array([rf_amp, 0]) ),
            'rx0_en': ( rx_t, np.array([1, 0]) )
            # 'leds': ( np.array([k*rf_interval]), np.array(k) )
            })
    expt.plot_sequence()
    plt.show()
    rxd, msgs = expt.run()
    
    #print(rxd)
    rxplot(rxd)
    


    expt.close_server(only_if_sim=True)

    expt._s.close() # close socket on client




if __name__ == "__main__":
    long_loopback(1000, 3) #
