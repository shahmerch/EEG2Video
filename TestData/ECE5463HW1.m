% Define the initial point coordinates
p = [1/sqrt(3), 1/sqrt(6), 1/sqrt(2)]';

% Rotation angles in degrees
theta_x = 30;   % Rotation around the x-axis
theta_y = 120;  % Rotation around the y-axis
theta_z = -120; % Rotation around the z-axis

% Convert angles to radians
theta_x = deg2rad(theta_x);
theta_y = deg2rad(theta_y);
theta_z = deg2rad(theta_z);

% Define rotation matrices
Rx = [1, 0, 0; 
      0, cos(theta_x), -sin(theta_x);
      0, sin(theta_x), cos(theta_x)];

Ry = [cos(theta_y), 0, sin(theta_y); 
      0, 1, 0;
      -sin(theta_y), 0, cos(theta_y)];

Rz = [cos(theta_z), -sin(theta_z), 0; 
      sin(theta_z), cos(theta_z), 0;
      0, 0, 1];

% Combined rotation matrix
R = Rz * Ry * Rx;

% Calculate the new coordinates of the point
p_prime = R * p;

% Display the results
disp('Coordinates of the new point p'' are:');
disp(p_prime);

disp('The rotation matrix R is:');
disp(R);
