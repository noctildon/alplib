# Welcome to ALPlib.

### This is a python library for performing physics calculations for axion-like-particles (ALPs).
### Examples and usage instructions incoming.


# Required tools
* python >3.7
    * numpy
    * scipy
    * mpmath
    * numba (coming soon)



# Classes and Methods

## Constants and Conventions
* Global constants (SM parameters, unit conversions, etc.) are stored in `constants.py` and have the naming convention `GLOBAL_CONSTANT_NAME`
* All units in alplib are in MeV, cm, kg, and s by default unless specifically stated, for example densities given in g/cm^2.

## The Material class
The `Material` class is a container for the physical constants and parameters pertaining to the materials used in experimental beam targets and detectors.
There are a number of material parameters stored in a JSON dictionary in `data/mat_params.json`, named according to the chemical name of the material, e.g. 'Ar' or 
'CsI'. One would initialize a detector or beam target, for instance, in the following way;
```
target_DUNE = Material('C')
det_DUNE = Material('Ar')
```
Further specifications for the volumes of the target/detector can also be specified for each instance that you may be interested in. The optional parameters
`fiducial_mass` (in kg)
```
det_DUNE = Material('Ar', fiducial_mass=50000)
```

## The AxionFlux super class
The class `fluxes.AxionFlux` is a super-class that can be inherited by any class that models a specific instance of a source of axion flux. It's most basic members are the `axion_energy` and `axion_flux` arrays, which together make a list of pairs of energies (in MeV) and event weights (in counts/second). Any class that inherits `AxionFlux` should populate `axion_energy` and `axion_flux` during its simulation routine - this flux class can then be passed to event generators (e.g. `fluxes.PhotonEventGenerator()`) to generate scattering or decay event weights at a detector module.

`AxionFlux` also has a default propagate method (which can be modified depending on the specific instance of the class inheriting `AxionFlux`) that looks at `AxionFlux.lifetime()` to propagate the flux weights to the detector. 



### Generators and Fluxes that inherit AxionFlux

### Detection Classes and Event Rates

## Production and Detection Cross Sections

## MatrixElement and Monte Carlo Methods
The super class `MatrixElement2` and its inheritors offers a way to embed any 2->2 scattering process 1 2 -> 3 4. One simply needs to input the masses `m1`, `m2`, `m3`, `m4`, and the `__call__` method will return the squared matrix element as a function of the Mandelstam variables `s` and `t`. Below we outline the monte carlo simulation algorithm for 2-to-2 scattering as an example;


For any $2\to 2$ process $\phi_1 (p_1) \phi_2 (p_2) \rightarrow \phi_3 (p_3) \phi_4 (p_4)$, we can express the Lorentz invariant amplitude as a function of Mandelstam $s, t$ as $\mathcal{A}(s,t) = \braket{\mid\mathcal{M}(s,t)\mid^2}$, for Mandelstam definitions
\begin{align}
    s &= (p_1 + p_2)^2 = (p_3 + p_4)^2 \\
    t &= (p_1 - p_3)^2 = (p_2 - p_4)^2 \\
    u &= (p_1 - p_4)^2 = (p_2 - p_3)^2 \\
    u &= m_1^2 + m_2^2 + m_3^2 + m_4^2 - s - t
\end{align}
Given masses $m_i$, $i=1,\dots,4$ and incoming 4-momenta $p_1^\mu$ and $p_2^\mu$, we can simulate the outgoing 4-momenta $p_3^\mu$ and $p_4^\mu$ as follows;

\begin{enumerate}
    \item Find the Lorentz boost velocity $v$ to transform $p_1^\mu$ and $p_2^\mu$ to the CM frame. In the special case of a fixed target with an incident beam along the $z$-direction, e.g. $p_1^\mu = (E_1, 0, 0, p_1)$ and $p_2^\mu = (m_2, 0, 0, 0)$, we have
        \begin{align}
            v &= \dfrac{p_1}{m_2 + E_1} \\
            \to \gamma &= (1 - v^2)^{-1/2}
        \end{align}
    \item In the CM frame, the energies of the outgoing particles are fixed, so $t$ is purely a function of $\cos\theta$;
        \begin{equation}
            t(\cos\theta^*) = m_1^2 + m_3^2 - 2E^*_1 E^*_3 + 2 p^*_1 p^*_3 \cos\theta^*
        \end{equation}
    where
        \begin{equation}
            \mid {p_{1,3}^*}\mid^2 = \dfrac{(s - m_{1,3}^2 - m_{2,4}^2)^2 - 4m_{1,3}^2 m_{2,4}^2}{4s}
        \end{equation}
        \begin{equation}
            E^*_{1,3} = \sqrt{\mid p_{1,3}^* \mid^2 - m_{1,3}^2}
        \end{equation}
    \item Draw $j=1,\dots,N$ random variates $u_j \sim U(-1,1)$ and find $t_j(u_j)$
    \item The differential scattering cross section in the CM frame is given by 
    \begin{equation}
        \dfrac{d\sigma}{dt} = \dfrac{1}{16\pi(s-(m_1 + m_2)^2)(s-(m_1 - m_2)^2)} \mid \mathcal{M}(s,t_j) \mid^2
    \end{equation}
    with
    \begin{equation}
        \dfrac{d\sigma}{d(\cos\theta^*)} = 2 p^*_1 p^*_3 \dfrac{d\sigma}{dt}
    \end{equation}
    Therefore, the MC weights $w_j$ are given by
    \begin{equation}
        w_j = \dfrac{2}{N} 2 p^*_1 p^*_3 \dfrac{d\sigma(s, t_j)}{dt}
    \end{equation}
    where the factor of $2/N$ is the MC volume factor.
    
    \item Boost to the lab frame, transforming the outgoing 4-momenta as
    \begin{align}
        E_3 &= \gamma (E_3^* + \beta p_3^* \cos\theta^*) \\
        p_3 \cos\theta &= \gamma (p_3^* \cos\theta^* + \beta E_3^*)
    \end{align}
    which implies
    \begin{align}
        \dfrac{d\sigma}{dE_3} &= \bigg( \dfrac{1}{\gamma \beta p_3^*}\bigg)\dfrac{d\sigma}{d(\cos\theta^*)} \nonumber \\
        &= \dfrac{2 p_1^*}{\gamma \beta} \dfrac{d\sigma}{dt}.
    \end{align}
    {\color{red}However, we need to multiply by the MC volume $E_3^{max} - E_3^{min} = 2\gamma \beta p_3^*$ and divide out the factor of 2 associated with the MC volume for the cosine integrated weights, leaving the differential energy weights simply as
    \begin{align}
        w_j &= \frac{1}{2}2\gamma\beta p_3^* \dfrac{d\sigma}{dE_3} \nonumber \\
        &= \dfrac{d\sigma}{d(\cos\theta^*)}
    \end{align}}
    More generally, the weights pickup a Jacobian factor $J_j$, depending on what variable we want to transform to~\cite{Catchen_78};
    \begin{align}
        J\bigg[\dfrac{p_3^*, \Omega^*}{p_3, \Omega}\bigg] &= \dfrac{p_3^2 E_3^*}{p_3^{*2} E_3} \\
        J\bigg[\dfrac{\Omega^*}{\Omega}\bigg] &= \bigg(\dfrac{v}{v^*}\bigg)^2 (\cos\theta \cos\theta^* + \sin\theta \sin\theta^* \cos(\phi - \phi^*)) \nonumber \\
        &= \bigg(\dfrac{v}{v^*}\bigg)^2 \cos\delta
    \end{align}
    Where $\delta$ is the angle between $\vec{v}$ and $\vec{v}^*$. The angles can be related through the Lorentz transformation for a simple boost along the $z$-axis;
    \begin{align}
        \theta &= \arccos \bigg\{\dfrac{\gamma (p_3^* \cos\theta^* + \beta E_3^*)}{\sqrt{(\gamma (E_3^* + \beta p_3^* \cos\theta^*))^2 - m_3^2)}}\bigg\} \\
        \phi &= \phi^*
    \end{align}
\end{enumerate}


## Decay Modes

## Crystal Scattering

# Examples
