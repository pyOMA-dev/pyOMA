import numpy as np
import matplotlib.pyplot as plt


def frequency_response(N, n_nodes, inp_node, fs, num_modes=None, out_quant='a', E=2.1e11, rho=7850, A=0.0343, L=200, zeta=0.05):
    '''
    Returns the onesided FRF matrix of the linear(ized) system
    at N//2 + 1 frequency lines for all nodes in meas_nodes
    by default the accelerance with input force at the last node is returned
    
    Uses numerically computed modal parameters and discrete system matrices
    The FRF may not be completely equivalent to analytical solutions
    
    inp_node is the ANSYS node number -> index is corresponding to
        meas_nodes (if compensated) or
        nodes_coordinates if not compensated
    '''
    assert ~ N % 2
    nodes_coordinates = np.linspace(0, L, n_nodes)
    
    nyq_omega = fs / 2 * 2 * np.pi
    num_modes_ = int((nyq_omega * L / np.pi / np.sqrt(E / rho) * 2 + 1) // 2)
    if num_modes is None:
        num_modes = num_modes_
    assert num_modes <= num_modes_
    
    j = np.arange(1, num_modes + 1, 1)
    frequencies = (2 * j - 1) / 2 * np.pi / L * np.sqrt(E / rho) / 2 / np.pi
    mode_shapes = np.sin((2 * j[np.newaxis, :] - 1) / 2 * np.pi / L * nodes_coordinates[:, np.newaxis])
    # modal mass verification
    # print(np.sum(mode_shapes**2*rho*A, axis=0)*L/n_nodes, rho*A*L/2)
    
    print(f'system has {num_modes} modes and frequencies {frequencies}')
    # plt.figure()
    # for i in range(num_modes):
        # plt.plot(mode_shapes[:, i], label=i)
    # plt.show()
    
    domega = 2 * np.pi * fs / N
    
    #omegas = np.linspace(0, nyq_omega, N // 2 + 1, True)
    omegas = np.fft.fftfreq(N, 1 / fs) * 2 * np.pi

    assert np.isclose(domega, omegas[1] - omegas[0])
    omegas = omegas[:, np.newaxis]
    
    omegans = frequencies * 2 * np.pi
    
    frf = np.zeros((N, n_nodes), dtype=complex)
    
    for mode in range(num_modes):
        
        omegan = omegans[mode]
        kappa = omegan**2
        mode_shape = mode_shapes[:, mode]
        modal_coordinate = np.abs(mode_shape[inp_node])
        
        if out_quant == 'a':
            frf += -omegan**2 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
        elif out_quant == 'v':
            frf += omegan / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
        elif out_quant == 'd':
            frf += 1 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
    # make the ifft real-valued
    #print(frf.imag[0, :], 'should be zero')
    frf.imag[0, :] = 0
    # print(frf[N // 2 , :],' should be real')
    frf[N // 2, :] = np.abs(frf[N // 2, :])
    # print(frf[N // 2 , :],' should be real')
    return omegas, frf


def ambient_ifrf(N, n_nodes, inp_nodes, fs, f_scale, seed=None, snr_db=np.inf, **kwargs):
    
    rng = np.random.default_rng(seed)
    
    # phase = rng.uniform(-np.pi, np.pi, (N // 2 + 1, n_nodes))
    # enforce same phase on all nodes, which is assumed in the analytical solution
    phase = np.repeat(rng.uniform(-np.pi, np.pi, (N // 2 + 1, 1)), n_nodes, axis=1)
    ampli = np.exp(1j * np.concatenate((phase[:N // 2, :], -1 * np.flip(phase[1:, :], axis=0))))
    Pomega = f_scale * np.ones((N, n_nodes), dtype=complex) * ampli
    
    # make the ifft real-valued
    Pomega.imag[0, :] = 0
    Pomega[N // 2, :] = np.abs(Pomega[N // 2, :])
    
    sig = np.zeros((N, n_nodes))
    # compute ifft for each combination of input and output node
    # use linear superposition of output signals from each input node
    
    for inp_node_ind, inp_node in enumerate(inp_nodes):
        _, this_frf = frequency_response(N, n_nodes, inp_node, fs, **kwargs)
        for channel in range(this_frf.shape[1]):
            this_sig = np.fft.ifft(this_frf[:, channel] * Pomega[:, inp_node_ind])
            assert np.all(np.isclose(this_sig.imag, 0))
            sig[:, channel] += this_sig.real
    
    time_values = np.linspace(1 / fs, N / fs, N) #  ansys also starts at deltat
    
    power_signal = np.mean(sig**2, axis=0)
    print(power_signal)
    snr = 10**(snr_db / 10)
    noise_power = power_signal / snr
    print(noise_power)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(noise_power), (N, n_nodes))
    sig += noise
    
    return time_values, sig


def ambient_spectral(N, n_nodes, inp_nodes, fs, f_scale, **kwargs):
    """
    assume inp_nodes==ref_nodes
    psd_matrix_shape = (n_nodes, inp_nodes, N)
    for each combination of nodes
        psd = H_i * conj(H_j) * ( f_scale**2 )?
        subtract mean?
    """
    
    psd_matrix = np.zeros((n_nodes, len(inp_nodes), N), dtype=complex)
    corr_matrix = np.zeros((n_nodes, len(inp_nodes), N))
    
    Y = np.zeros((N, n_nodes), dtype=complex)
    for inp_node_ind, inp_node in enumerate(inp_nodes):
        omegas, frf = frequency_response(N, n_nodes, inp_node, fs, **kwargs)
        # assume same phase and amplitude on all input nodes
        Y += frf * f_scale
        
    for node in range(n_nodes):
        for inp_node_ind, inp_node in enumerate(inp_nodes):
            
            psd = Y[:, node] * np.conj(Y[:, inp_node])
            # assert np.all(np.isclose(psd.imag, 0))
            psd = psd.real
            
            psd_matrix[node, inp_node_ind, :] = psd #/ N * fs
            
            # analytical solution for convolution difficult, use numerical inverse of analytical PSD
            corr = np.fft.ifft(psd)
            assert np.all(np.isclose(corr.imag, 0))
            corr_matrix[node, inp_node_ind, :] = corr.real / N
    
    lags = np.linspace(0, N / fs, N, False)
    
    #analytical SDOF solution
    # zeta = 0.05
    # omega = 40.62231788528593
    # k = omega**2
    # psdan = omega**4/ (k**2 * (1 + (4 * zeta**2 - 2) * (omegas / omega)**2 + (omegas / omega)**4)) * f_scale**2
    # corran = np.fft.ifft(psdan[:, 0]).real
    
    return omegas, lags, psd_matrix, corr_matrix

if __name__ == '__main__':
    
    def fplot(self, x,y,**kwargs):
        return self.plot(np.fft.fftshift(x), np.fft.fftshift(y), **kwargs)
    plt.fplot = fplot
    import matplotlib.axes
    matplotlib.axes.Axes.fplot = fplot
    
    N = 2**15
    n_nodes = 10
    inp_nodes = [6, 9]
    fs = 128
    f_scale = 1
    
    # omegas, frf = frequency_response(N, n_nodes, inp_nodes[0], fs)
    # plt.figure()
    # plt.plot(omegas, np.abs(frf))
    # plt.show()
    
    time_values, sig = ambient_ifrf(N, n_nodes, inp_nodes, fs, f_scale, snr_db=np.inf)
    # fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    # for i in range(9):
        # axes.flat[i].plot(time_values, sig[:, i + 1])
    # plt.show()
    omegas, frf = frequency_response(N, n_nodes, inp_nodes[0], fs, None)
    omegas, taus, psd, corr = ambient_spectral(N, n_nodes, inp_nodes, fs, f_scale)
    
    #analytical SDOF solution
    # zeta = 0.05
    # omega = 40.62231788528593
    # k = omega**2
    # psdan = omega**4 / (k**2 * (1 + (4 * zeta**2 - 2) * (omegas / omega)**2 + (omegas / omega)**4)) * f_scale**2
    # corran = np.fft.ifft(psdan[:, 0]).real

    
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    for j, inp_node in enumerate(inp_nodes):
        sig1 = sig[:, inp_node]
        for i in range(9):
            sig2 = sig[:, i + 1]
            psd_num = np.fft.fft(sig1) * np.conj(np.fft.fft(sig2)) #/ N * fs
        # sum_psd = np.sum(np.abs(psd[i , :, :]), axis=0)
        # axes.flat[i].fplot(omegas, sum_psd, alpha=0.5)
            axes.flat[i].fplot(omegas, np.abs(psd[i + 1, j, :]), alpha=0.5, marker='x')
            axes.flat[i].fplot(omegas, np.abs(psd_num), alpha=0.5, marker='+')
            # axes.flat[i].fplot(omegas, psdan, alpha=0.5)
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
        
    for j, inp_node in enumerate(inp_nodes):
        sig1 = sig[:, inp_node]
        sig1_pad = np.concatenate((np.zeros(N - 1), sig1))
        for i in range(9):
            sig2 = sig[:, i + 1]
            corr_num = np.correlate(sig2, sig1_pad, 'valid') / N
            # print(corr_num, corr_num.shape)
            # sum_corr = np.sum(corr[i+1, :, :], axis=0)
            axes.flat[i].plot(taus, corr[i + 1, j, :], alpha=0.5, marker='x')
            axes.flat[i].plot(taus, corr_num, alpha=0.5, marker='+')
            axes.flat[i].axhline(np.mean(sig1 * sig2))
            # # axes.flat[i].plot(taus, corran, alpha=0.5)
    axes.flat[i].set_xlim((-0.1, 2.5))
    # axes.flat[i].set_ylim((-7, 7))
    # plt.figure()
    for ref_ind, ref_node in enumerate(inp_nodes):
    
        ratios = []
        for node in range(n_nodes):
            power = np.sum(sig[:, node] * sig[:, ref_node])
            print(f'Power time-domain: {power}')
            print(f'Theoretic powers')
            power_psd = np.mean(psd[node, ref_ind, :])
            assert np.isclose(power_psd.imag, 0)
            print(f'PSD: {power_psd.real}')
            power_corr = corr[node,ref_ind,0] * N
            print(f'0-lag corr: {power_corr}')
            ratios.append(power_corr/power)
            print(f'ratio: {power/power_corr}')
            print('\n')
        # plt.plot(ratios)
        # plt.axvline(ref_node)
        # plt.axhline(1, color='red', alpha=0.1)
    plt.show()