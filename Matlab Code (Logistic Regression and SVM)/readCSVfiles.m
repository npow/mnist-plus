clear all; close all; clc

%% import train csv data

trainX = importdata('train_inputs.csv');
% ID, 2304 dim
trainX=trainX.data(:,2:end);

trainy = importdata('train_outputs.csv');
% ID, 0-9
trainy=trainy.data(:,2);

%% import test csv data
testX = importdata('test_inputs.csv');
% ID, 2304 dim
testX=testX.data(:,2:end);

%% for conv NN
trainX=reshape(trainX',48,48,50000);
train_y=zeros(10,50000);
ind_y = sub2ind([10 50000],trainy+1,(1:50000)');
train_y(ind_y)=1;

% test_x=trainX(:,:,1:10000);
% test_y=train_y(:,1:10000);
% 
% train_x=trainX(:,:,10001:end);
% train_y=train_y(:,10001:end);

% testX=reshape(testX',48,48,20000);

%% pre-Processing (low-pass)
close all
%Determine good padding for Fourier transform
PQ = paddedsize([48 48]);
train_x=zeros(28,28,50000);
% test_x=zeros(28,28,20000);

%Create a Gaussian Lowpass filter 8% the width of the Fourier transform
D0 = 0.2*PQ(1);
L = lpfilter('gaussian', PQ(1), PQ(2), D0);

for i=1:50
% for i=1:20000
    figure
    subplot(131)
%     trainX1(:,:,i)=trainX(trainX(:,:,i)>0.2,:,i);
    imagesc(trainX(:,:,i))
    title(num2str(trainy(i)))
    colormap gray
    axis square
% Calculate the discrete Fourier transform of the image
    F=fft2(trainX(:,:,i),size(L,1),size(L,2));
%     F=fft2(testX(:,:,i),size(L,1),size(L,2));
    
    % Apply the lowpass filter and the highpass filter to the Fourier spectrum of the image
    LPFS_football = L.*F;   
    
    % convert the result to the spacial domain.
    LPF_football=real(ifft2(LPFS_football)); 
    
    % Crop the image to undo padding
    LPF_football=LPF_football(1:48, 1:48);
    subplot(132)
    imagesc(LPF_football)
    axis square
     train_x(:,:,i) = imresize(LPF_football, 7/12);
%     test_x(:,:,i) = imresize(LPF_football, 7/12);
    subplot(133)
    imagesc(train_x(:,:,i))
    axis square
    i
end

%% pre-Processing (down sample)
figure
subplot(221), imshow(HPF_football, [])
B = imresize(HPF_football, 7/12);
subplot(222), imshow(B, [])

%% generate output csv file
randy = importdata('test_output_random.csv');
% ID, 0-9
randy=randy.data;

randy(:,2)=h-1;
randy=num2cell(randy);
randy=[{'Id', 'Prediction'}; randy];

cell2csv('test_output_cNN1000.csv',randy,','); % convert to csv file

%% logistic regression

B = mnrfit(trainX,trainy+1);

%%
prob = mnrval(B,trainx);

%%
yhat = find(prob==max(prob,2));


