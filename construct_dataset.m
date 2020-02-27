function [exp_data]=construct_dataset(db_name,num_test,topNum)
addpath('./DB/');
%%choose and load dataset
if strcmp(db_name,'VOC2007&Caltech101')    
load VOC2007 %source data
Xs = double(data(:,1:end-1));
Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
ys = data(:,end);
clear data; %clear labels;

load Caltech101 %target data
Xt = double(data(:,1:end-1));
Xt = normalize1(Xt);%Xt/max(max(abs(Xt)));
yt = data(:,end);
clear data; %clear labels;

elseif strcmp(db_name, 'Caltech256&ImageNet')
    load dense_imagenet_decaf7_subsampled; %target data
    Xt = normalize1(fts);
    Xt = double(Xt);
    yt = double(labels);
    clear fts; clear labels;

    load dense_caltech256_decaf7_subsampled; %source data
    Xs = normalize1(fts);%fea;
    Xs = double(Xs);
    ys = double(labels);
    clear fts; clear labels;
 elseif strcmp(db_name,'MNIST&USPS')   
   load MNIST_vs_USPS.mat %source data
   Xs = double(X_src)';
   Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
   ys = double(Y_src); %vector label
   Xt = double(X_tar)';
   Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
   yt = double(Y_tar); %vector label
   clear X_src;  clear Y_src;  clear X_tar;  clear Y_tar;

end
 
[ndatat,~]      =     size(Xt);
R               =     randperm(ndatat);
test            =     Xt(R(1:num_test),:);
ytest           =     yt(R(1:num_test));
R(1:num_test)   =     [];
train           =     Xt(R,:);
yt              =     yt(R);
train_ID = R;


ytnew = knnclassify(train,Xs,ys);
acc = length(find(ytnew==yt))/length(yt)
num_train       =     size(train,1);
if topNum == 0
    topNum      =     round(0.02*num_train);%set top Num as Two percent of  the number of train

end
DtrueTestTrain  =    distMat(test,train);
[~,idx]         =    sort(DtrueTestTrain,2);
idx             =    idx(:,1:topNum);
WtrueTestTrain  =    zeros(num_test,num_train);
for i=1:num_test
    WtrueTestTrain(i,idx(i,:)) =1;
end

YS            =  repmat(ys,1,length(ytest));
YT            =  repmat(ytest,1,length(ys));
WTT           =  (YT==YS');


XX=[Xs;Xt];
samplemean              = mean(XX,1);
Xs                      = Xs-repmat(samplemean,size(Xs,1),1);
train                   = train-repmat(samplemean,size(train,1),1);
test                    = test-repmat(samplemean,size(test,1),1);


exp_data.Xs         =   Xs ;
exp_data.test       =   test;
exp_data.train      =   train;
exp_data.ytnew      =   ytnew ;
exp_data.yt      =   yt;
exp_data.ys         =   ys ;
exp_data.WTT           =WTT ;
end
