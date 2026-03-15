clc
clear
close all
set(0,'DefaultFigureWindowStyle','docked')

g = 9.81; % gravitational acceleration in mm/s^2
L1 = 10; % lateral length of rectangle in mm
L2 = 10; % depth of rectangle in mm

n = 200; % number of datapoints/measurements to be generated
training_data=[];

for i=1:n
    motor1_displacement = 10*rand; %in mm
    motor2_displacement = 10*rand; %in mm
    motor3_displacement = 10*rand; %in mm
    motor4_displacement = 10*rand; %in mm
    % Corner coordinates of the rectangle:
    corners = [ 0 0 motor1_displacement;
                L1 0 motor2_displacement;
                L1 L2 motor3_displacement;
                0 L2 motor4_displacement];
    [accelerometer_reading, eul_rad, eul_deg, R_b2w] = compute_accel_from_corners(corners, g);
    new_training_data = [accelerometer_reading',motor1_displacement,motor2_displacement,motor3_displacement,motor4_displacement];
    training_data=[training_data;new_training_data];
end

% Create a table with headers
T = array2table(training_data, ...
    'VariableNames', {'ax','ay','az','motor1_diplacement[mm]','motor2_diplacement[mm]','motor3_diplacement[mm]','motor4_diplacement[mm]'});
% Write to CSV in current folder
csv_filename = 'accelerometer_data_for_training.csv';
writetable(T, csv_filename);

function [a_body, eul_rad, eul_deg, R_b2w] = compute_accel_from_corners(corners, g)
% compute_accel_from_corners  Compute accelerometer reading and roll/pitch/yaw
%
% USAGE:
%   [a_body, eul_rad, eul_deg, R_b2w] = compute_accel_from_corners(corners)
%   [a_body, eul_rad, eul_deg, R_b2w] = compute_accel_from_corners(corners, g)
%
% INPUT:
%   corners : 4x3 matrix. Each row = [x y z] of corners A,B,C,D in order
%             around the rectangle (e.g., clockwise A->B->C->D).
%   g       : (optional) gravity magnitude in m/s^2. Default g = 9.81.
%             Gravity is assumed to point along negative world-$z$:
%             $g_{world} = [0;0;-g]$.
%
% OUTPUT:
%   a_body  : 3x1 accelerometer reading in the platform (body) frame
%             [ax; ay; az] in m/s^2 (what the accelerometer at platform center
%             would measure when static in gravity).
%   eul_rad : 3x1 Euler angles [roll; pitch; yaw] in radians.
%   eul_deg : 3x1 Euler angles [roll; pitch; yaw] in degrees.
%   R_b2w   : 3x3 rotation matrix from body frame to world frame.
%
% CONVENTION:
%   Euler angles follow Z-Y-X intrinsic (yaw -> pitch -> roll), returned as
%   [roll; pitch; yaw]. That is, R_b2w = Rz(yaw)*Ry(pitch)*Rx(roll).
%
% EXAMPLE:
%   corners = [0 0 0; 1 0 0; 1 2 0.1; 0 2 0.1];   % slight tilt
%   [a, e_rad, e_deg] = compute_accel_from_corners(corners)

    if nargin < 2 || isempty(g)
        g = 9.81;
    end

    % Validate input
    if ~ismatrix(corners) || size(corners,1) ~= 4 || size(corners,2) ~= 3
        error('corners must be a 4x3 matrix: rows = [x y z] for corners A,B,C,D.');
    end

    % Extract corner points (rows)
    A = corners(1, :)';
    B = corners(2, :)';
    C = corners(3, :)';
    D = corners(4, :)';

    % Build primary in-plane axes using edges A->B and A->D
    v1 = B - A;   % candidate for body x-axis
    v2 = D - A;   % candidate for body y-axis

    if norm(v1) < eps || norm(v2) < eps
        error('Corner spacing too small or duplicate corners detected.');
    end

    x_axis = v1 / norm(v1);

    % Make v2 orthogonal to x_axis (Gram-Schmidt)
    v2_orth = v2 - (dot(v2, x_axis) * x_axis);
    if norm(v2_orth) < 1e-8
        error('Provided corners produce nearly collinear edges. Make sure corners are ordered properly around the rectangle.');
    end
    y_axis = v2_orth / norm(v2_orth);

    % z axis as plane normal (right-handed)
    z_axis = cross(x_axis, y_axis);
    z_norm = norm(z_axis);
    if z_norm < 1e-8
        error('Corners are degenerate; cannot compute a well-defined normal.');
    end
    z_axis = z_axis / z_norm;

    % Construct rotation matrix from body to world (columns are body axes in world coords)
    R_b2w = [x_axis, y_axis, z_axis];   % 3x3

    % World to body rotation
    R_w2b = R_b2w';   % orthonormal => inverse = transpose

    % Gravity in world coordinates: default pointing down along world -Z
    g_world = [0; 0; -g];

    % Accelerometer reading in body frame (what a 3-axis accelerometer would measure)
    a_body = R_w2b * g_world;

    % --- Euler angles extraction (Z-Y-X intrinsic: yaw, pitch, roll) ---
    r = R_b2w;  % body-to-world rotation matrix
    % Elements
    r11 = r(1,1); r12 = r(1,2); r13 = r(1,3);
    r21 = r(2,1); r22 = r(2,2); r23 = r(2,3);
    r31 = r(3,1); r32 = r(3,2); r33 = r(3,3);

    % pitch (theta)
    theta = asin( clamp(-r31, -1.0, 1.0) );  % -r31 = sin(theta)
    % To avoid gimbal issues where cos(theta) ~ 0:
    cos_theta = cos(theta);

    tol = 1e-6;
    if abs(cos_theta) > tol
        % normal case
        psi = atan2(r21, r11);        % yaw
        phi = atan2(r32, r33);        % roll
    else
        % Gimbal lock: pitch ~ +/-90 deg. Set roll = 0 and compute yaw differently.
        % See standard fallback formulas.
        phi = 0;
        if r31 <= -1  % theta = +pi/2
            psi = atan2(-r12, r22);
        else          % theta = -pi/2
            psi = atan2(-r12, r22);
        end
    end

    eul_rad = [phi; theta; psi];          % [roll; pitch; yaw] in radians
    eul_deg = rad2deg(eul_rad);           % in degrees

    % Display results
    fprintf('Accelerometer reading at platform center (body frame) [ax; ay; az] (m/s^2):\n');
    fprintf('  [%.6f; %.6f; %.6f]\n', a_body(1), a_body(2), a_body(3));
    fprintf('Euler angles (ZYX intrinsic convention):\n');
    fprintf('  roll (deg)  = %.4f\n', eul_deg(1));
    fprintf('  pitch(deg)  = %.4f\n', eul_deg(2));
    fprintf('  yaw  (deg)  = %.4f\n', eul_deg(3));
end

% Helper: clamp to numerical range
function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end
