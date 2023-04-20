%% Visualization of three point correspondences in both images
im1 = imread('im1.jpg');
im2 = imread('im2.jpg');

% Points L, M, N (corners of the book) in image 1
lmn1 = 1.0e+03 * [1.3715 1.0775;
                  1.8675 1.0575;
                  1.3835 1.4415];

% Points L, M, N (corners of the book) in image 2
lmn2 = 1.0e+03 * [1.1555 1.0335;
                  1.6595 1.0255;
                  1.1755 1.3975];

% Annotate and show images  
figure;
subplot(2,1,1);imshow(im1);hold on;
plot(lmn1(:,1),lmn1(:,2),'c+','MarkerSize',10);
labels={'L','M','N'};
for i=1:length(labels)
    ti=text(lmn1(i,1),lmn1(i,2),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
end
subplot(2,1,2);imshow(im2);hold on;
plot(lmn2(:,1),lmn2(:,2),'c+','MarkerSize',10);
labels={'L','M','N'};
for i=1:length(labels)
    ti=text(lmn2(i,1),lmn2(i,2),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
end


%% Your task is to implement the missing function 'trianglin.m'
% The algorithm is described in the lecture slides.
% Output should be the homogeneous coordinates of the triangulated point.

% Load the pre-calculated projection matrices
load P1.mat
load P2.mat

% Triangulate each corner
L = trianglin(P1,P2,[lmn1(1,:) 1],[lmn2(1,:) 1]);
M = trianglin(P1,P2,[lmn1(2,:) 1],[lmn2(2,:) 1]);
N = trianglin(P1,P2,[lmn1(3,:) 1],[lmn2(3,:) 1]);

% We can then compute the width and height of the picture on the book cover
% Convert the above points to cartesian, form vectors corresponding to
% book covers horizontal and vertical sides using the points and calculate 
% the norm of these to acquire the height and width (mm).
%%-your-code-starts-here-%%
picture_width_mm = 0
picture_height_mm = 0
%%-your-code-ends-here-%%