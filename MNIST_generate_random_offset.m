clear all
close all
clc
tic
%load MNIST_dataset.mat;
%For Windows:
%addpath('C:\Users\User\Dropbox\Dropbox\ml_projects\ML_summer\matconvnet-master\data\mnist')
%For Ubuntu
addpath('/home/rob/Dropbox/ml_projects/ML_summer/matconvnet-master/data/mnist')

%Load all the data
train_data = loadMNISTImages('train-images-idx3-ubyte');
train_classlabel = loadMNISTLabels('train-labels-idx1-ubyte');
test_data = loadMNISTImages('t10k-images-idx3-ubyte');
test_classlabel = loadMNISTLabels('t10k-labels-idx1-ubyte');

X_train = train_data';
X_test = test_data';
y_train = train_classlabel;
y_test = test_classlabel;

clearvars test_data train_data test_classlabel train_classlabel

%Start the random tiling

im_size = 110;  %What size must the total image be?
tile_size = 28;  %tile size of input tile, usually 28 for MNIST

Ntest = size(X_test,1);

X_test_tiled = zeros(Ntest,im_size^2);

%loop over Ntest to give every tile a random offset in the image
for i = 1:Ntest
    im = reshape(X_test(i,:),tile_size,tile_size);
    im_tiled = zeros(im_size,im_size);
    
    %Generate random offsets
    offsets = randi([1 (im_size-tile_size)],2,1);
    
    %Put the tile in the image
    im_tiled(offsets(1):offsets(1)+tile_size-1,offsets(2):offsets(2)+tile_size-1) = im;
    
    %Back to rowvector and place in file
    X_test_tiled(i,:) = reshape(im_tiled,1,im_size^2);
end
toc
%%
csvwrite('X_train.csv',X_train)
csvwrite('y_train.csv',y_train)
csvwrite('X_test.csv',X_test)
csvwrite('y_test.csv',y_test)
csvwrite('X_test_tiled.csv',X_test_tiled)