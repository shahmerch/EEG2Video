% SEPIC Converter Simulation Parameters
Vin = 25;          % Input voltage in Volts
Vout = 30;         % Output voltage in Volts
Pout = 50;         % Output power in Watts
fs = 150e3;        % Switching frequency in Hz
D = 6/11;          % Duty cycle
L1 = 1e-3;         % Inductor L1 in Henry
L2 = 1.5e-3;       % Inductor L2 in Henry
C1 = 500e-6;       % Coupling capacitor in Farads
Co = 500e-6;       % Output capacitor in Farads
Rload = Vout^2 / Pout;  % Load resistance in Ohms
T = 1/fs;          % Switching period in seconds

% Simulation time
sim_time = 0.5e-3; % 0.5 milliseconds

% Create time vector
dt = T/1000;       % Time step
t = 0:dt:sim_time;

% Initialize variables
iL1 = zeros(size(t));
iL2 = zeros(size(t));
vL1 = zeros(size(t));
vL2 = zeros(size(t));
switch_status = zeros(size(t));

% Initial conditions
iL1(1) = 2;        % Initial current in L1
iL2(1) = 1.6667;   % Initial current in L2

% Simulation loop
for k = 1:length(t)-1
    % Determine switch status
    if mod(t(k), T) < D*T
        % Switch is ON
        switch_status(k) = 1;
        vL1(k) = Vin;
        vL2(k) = -Vin;
    else
        % Switch is OFF
        switch_status(k) = 0;
        vL1(k) = -Vout;
        vL2(k) = Vout;
    end
    
    % Update inductor currents
    iL1(k+1) = iL1(k) + (vL1(k)/L1)*dt;
    iL2(k+1) = iL2(k) + (vL2(k)/L2)*dt;
end

% Remove the last value for plotting consistency


% Plotting the waveforms
figure;
subplot(5,1,1);
plot(t*1e6, switch_status);
xlabel('Time (\mus)');
ylabel('Switch Status');
ylim([-0.1 1.1]);
xlim([[0 20]])
title('Switch Status');

subplot(5,1,2);
plot(t*1e6, iL1);
xlabel('Time (\mus)');
ylabel('i_{L1} (A)');
xlim([[0 20]])

title('Current through Inductor L1');

subplot(5,1,3);
plot(t*1e6, iL2);
xlabel('Time (\mus)');
ylabel('i_{L2} (A)');
xlim([[0 20]])

title('Current through Inductor L2');

subplot(5,1,4);
plot(t*1e6, vL1);
xlabel('Time (\mus)');
ylabel('v_{L1} (V)');
xlim([[0 20]])

title('Voltage across Inductor L1');

subplot(5,1,5);
plot(t*1e6, vL2);
xlabel('Time (\mus)');
ylabel('v_{L2} (V)');
xlim([[0 20]])

title('Voltage across Inductor L2');
