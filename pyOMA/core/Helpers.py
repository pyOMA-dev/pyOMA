'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2021  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created on 08.03.2021

@author: womo1998
'''
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10 ** sig_fig) == int(b * 10 ** sig_fig)
            )


def simplePbar(total):
    '''
    Divide the range of total in 100 discrete steps
        if total < 100 i.e. stepsize > 1: some steps may occur twice, printout must occur multiple times
        if total > 100 i.e. stepsize < 1: there are gaps, where no printout occurs
    For each call raise the step value by stepsize until step = total*100/total
    Check if the step is approximately 100
    
    For the last call, additionally a carriage return must be printed
    
    '''
    stepsize = 100 / total
    last = 0
    ncalls = 0
    while True:
        ncalls += 1
        while ncalls * stepsize // 1 > last:
            if logger.isEnabledFor(logging.INFO): print('.', end='', flush=True)
            last += 1
        if ncalls == total:  # np.isclose(step, 100):
            if logger.isEnabledFor(logging.INFO): print('', end='\n', flush=True)
        yield


def calc_xyz(az, elev, r=1):
    if np.abs(az) > 2 * np.pi:
        print('You probably forgot to convert to radians ', az)
    if np.abs(elev) > 2 * np.pi:
        print('You probably forgot to convert to radians ', elev)
    # for elevation angle defined from XY-plane up
    x = r * np.cos(elev) * np.cos(az)
    # x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis
    # down
    # for elevation angle defined from XY-plane up
    y = r * np.cos(elev) * np.sin(az)
    # y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
    z = r * np.sin(elev)  # for elevation angle defined from XY-plane up
    # z=r*np.cos(elev)# for elevation angle defined from Z-axis down

    # correct numerical noise
    for a in (x, y, z):
        if np.allclose(a, 0):
            a *= 0

    return x, y, z


def validate_array(arr):
    '''
    Determine whether the argument has a numeric datatype and if
    not convert the argument to a scalar object or a list.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.
    
    '''
    if arr is None:
        return None
    _NUMERIC_KINDS = set('buifc')
    if not arr.shape:
        return arr.item()
    elif arr.dtype.kind in _NUMERIC_KINDS:
        return arr
    else:
        return list(arr)


def get_method_dict():
    from pyOMA.core.PLSCF import PLSCF
    from pyOMA.core.PRCE import PRCE
    from pyOMA.core.SSICovRef import BRSSICovRef, PogerSSICovRef
    from pyOMA.core.SSIData import SSIData, SSIDataMC
    from pyOMA.core.VarSSIRef import VarSSIRef
    method_dict = {'Reference-based Covariance-Driven Stochastic Subspace Identification': BRSSICovRef,
                   'Reference-based Data-Driven Stochastic Subspace Identification': SSIDataMC,
                   'Stochastic Subspace Identification with Uncertainty Estimation': VarSSIRef,
                   'Poly-reference Least Squares Complex Frequency': PLSCF,
                   'Poly-reference Complex Exponential': PRCE, }
    return method_dict


def rq_decomp(a, mode='reduced'):
    q, r = np.linalg.qr(np.flipud(a).T, mode=mode)
    return np.flipud(r.T), q.T


def ql_decomp(a, mode='reduced'):
    q, r = np.linalg.qr(np.fliplr(a), mode)
    return q, np.fliplr(r)


def lq_decomp(a, mode='reduced', unique=True):
    '''
    Parameters
    ----------
    a : array_like, shape (..., M, N)
        An array-like object with the dimensionality of at least 2.
    mode : {'reduced', 'complete', 'r', 'raw'}, optional, default: 'reduced'
        If K = min(M, N), then

        * 'reduced'  : returns Q, R with dimensions (..., M, K), (..., K, N)
        * 'complete' : returns Q, R with dimensions (..., M, M), (..., M, N)
        * 'r'        : returns R only with dimensions (..., K, N)
    unique: bool
        "The QR decomposition is unique up to a sign change. Uniqueness can be
        enforced by constraining the diagonal elements of the R part to positive
        values." [Doehler, 2011]
    '''
    assert mode in ['reduced', 'complete', 'r', 'full']
    if mode == 'r':
        r = np.linalg.qr(a.T, mode)
    else:
        q, r = np.linalg.qr(a.T, mode)

    if unique:
        fact = np.sign(np.diag(r))
        r *= np.repeat(np.reshape(fact, (r.shape[0], 1)), r.shape[1], axis=1)
        if mode != 'r':
            q *= fact
            # print(np.allclose(a.T,q.dot(r)))

    if mode == 'r':
        return r.T
    else:
        return r.T, q.T


def calculateMAC(v1, v2):
    '''
    expects modeshapes in columns of v1 and/or v2
    outputs mac:
    ..math::

        \begin{bmatrix}
        MAC(v1[:,0],v2[:,0]) &   MAC(v1[:,0],v2[:,1]\\
        MAC(v1[:,1],v2[:,0]) &   MAC(v1[:,1],v2[:,1]
        \\end{bmatrix}

    '''

    v1_norms = np.einsum('ij,ij->j', v1, v1.conj())
    v2_norms = np.einsum('ij,ij->j', v2, v2.conj())
    MAC = np.abs(np.dot(v1.T, v2.conj())) ** 2 \
        / np.outer(v1_norms, v2_norms)

    return MAC.real


def calculateMPC(v):

    MPC = np.abs(np.sum(v ** 2, axis=0)) ** 2 \
        / np.abs(np.einsum('ij,ij->j', v, v.conj())) ** 2

    return MPC


def calculateMPD(v, weighted=True, regression_type='usv'):
    assert regression_type in ['ortho', 'arithm', 'usv']
    if regression_type == 'ortho':
        # orthogonal regression through origin
        # http://mathforum.org/library/drmath/view/68362.html
        real_ = np.real(v).copy()
        imag_ = np.imag(v).copy()
        ssxy = np.einsum('ij,ij->j', real_, imag_)
        ssxx = np.einsum('ij,ij->j', real_, real_)
        ssyy = np.einsum('ij,ij->j', imag_, imag_)

        MP = np.arctan2(2 * ssxy, (ssxx - ssyy)) / 2

        # rotates complex plane by angle MP
        v_r = v * np.exp(-1j * MP)  # (np.cos(-MP)+1j*np.sin(-MP))
        # calculates phase in range -180 and 180
        phase = np.angle(v_r, True)

        # rotates into 1st and 4th quadrant
        phase[phase > 90] -= 180
        phase[phase < -90] += 180
        # calculates standard deviation

        if not weighted:
            MPD = np.std(phase, axis=0)
        else:
            MPD = np.sqrt(
                np.average(
                    np.power(
                        phase -
                        np.mean(
                            phase,
                            axis=0),
                        2),
                    weights=np.absolute(v_r),
                    axis=0))

        # print(np.mean(phase, axis=0), np.sqrt(
        # np.mean(np.power(phase, 2), axis=0)), np.std(phase, axis=0), MPD)

        MP *= 180 / np.pi

    elif regression_type == 'arithm':
        phase = np.angle(v, True)

        phase[phase < 0] += 180

        if not weighted:
            MP = np.mean(phase, axis=0)
        else:
            MP = np.average(phase, weights=np.absolute(v), axis=0)

        if not weighted:
            MPD = np.std(phase, axis=0)
        else:
            MPD = np.sqrt(
                np.average(
                    np.power(
                        phase - MP,
                        2),
                    weights=np.absolute(v),
                    axis=0))

    elif regression_type == 'usv':

        MP = np.zeros(v.shape[1])
        MPD = np.zeros(v.shape[1])

        for k in range(v.shape[1]):
            mode_shape = np.array(
                [np.array(v[:, k]).real, np.array(v[:, k]).imag]).T

            _, _, V_T = np.linalg.svd(mode_shape, full_matrices=False)
            # print(U.shape,S.shape,V_T.shape)
            numerator = []
            denominator = []

            import warnings
            for j in range(len(v[:, k])):
                v[j, k]
                V_T[1, 1]
                V_T[1, 0]
                V_T[0, 1]
                V_T[1, 1]
                if weighted:
                    weight = np.abs(v[j, k])
                else:
                    weight = 1
                numerator_i = weight * np.arccos(np.abs(V_T[1, 1] * np.array(v[j, k]).real - V_T[1, 0] * np.array(
                    v[j, k]).imag) / (np.sqrt(V_T[0, 1] ** 2 + V_T[1, 1] ** 2) * np.abs(v[j, k])))
                warnings.filterwarnings("ignore")
                # when the arccos function returns NaN, it means that the value should be set 0
                # the RuntimeWarning might occur since the value in arccos
                # can be slightly bigger than 0 due to truncations
                if np.isnan(numerator_i):
                    numerator_i = 0
                numerator.append(numerator_i)
                denominator.append(weight)

            MPD[k] = np.degrees((sum(numerator) / sum(denominator)))
            MP[k] = np.degrees(np.arctan(-V_T[1, 0] / V_T[1, 1]))
            # MP in [-pi/2, pi/2] = [-90, 90]
            # phase=np.angle(v[:,k]*np.exp(-1j*np.radians(MP[k])),True)
            # print(np.mean(phase))
            # phase[phase>90]-=180
            # phase[phase<-90]+=180
            # print(np.mean(phase),np.sqrt(np.mean(phase**2)),np.std(phase),MPD[k])

    MP[MP < 0] += 180  # restricted to +imag region
    MPD[MPD < 0] *= -1

    # MP [0,180]
    # MPD >= 0
    return MPD, MP
