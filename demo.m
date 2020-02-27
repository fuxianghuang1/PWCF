function [recall, precision, mAP, rec, pre, retrieved_list] = demo(exp_data, param, method)

WtrueTestTraining = exp_data.WTT ;
pos               = param.pos;
r                 = param.r;
Xs                =    exp_data.Xs;
Xt                =    exp_data.train;
ys                =    exp_data.ys;
yt                =    exp_data.ytnew;
test              =    exp_data.test;

%% set parameters
setting.record = 0; %
setting.mxitr  = 10;
setting.xtol = 1e-5;
setting.gtol = 1e-5;
setting.ftol = 1e-8;
paras.k               = 50;
paras.sigma           = 0.4;
paras.m            =   0.3;
paras.theta1       =   10;
paras.theta2       =   10;
paras.lambda1      =   1;
paras.lambda2      =   10;
paras.lambda3      =   1e4;
paras.max_iter     =   50;
[paras.nt,paras.d] =   size(Xt);
[paras.ns,paras.d] =   size(Xs);
%% leraning
fprintf('......%s start ......\n\n', 'PWCF');
X=[Xs;Xt];
X1=[Xs;Xt;Xt];
N = size(X1,1);

[vec,val]  =   eig(X'*X);
[~,Idx]      =   sort(diag(val),'descend');
W          =   vec(:,Idx(1:r));
clear Idx;clear vec; clear val;

%%Construct triples
YS            =  repmat(ys,1,length(ys));
S             =  (YS==YS');
%[HS,HT,H]     =  fea_trans(Xt',Xs',yt,ys,paras);
Ds           =  EuDist2(Xs,Xs);
Dp            =  S.*Ds;
[~,Ip]        =  max(Dp,[],2);
Xp =[];
for i=1:length(ys)
    Xp         = [Xp; Xs(Ip(i),:)];%Similar sample
end
Dn            =  Ds-Dp;
[~,In]        =  min(Dn,[],2);
Xn =[];
for i= 1:length(ys)
    Xn        = [Xn; Xs(In(i),:)];
end

YT            =  repmat(yt,1,length(yt));
S             =  (YT==YT');
%[HS,HT,H]     =  fea_trans(Xt',Xs',yt,ys,paras);
Dt           =  EuDist2(Xt,Xt);
Dp            =  S.*Dt;
[~,Ip]        =  max(Dp,[],2);
for i=1:length(yt)
    Xp         = [Xp; Xt(Ip(i),:)];%Similar sample
end
Dn            =  Dt-Dp;
[~,In]        =  min(Dn,[],2);
for i= 1:length(yt)
    Xn        = [Xn; Xt(In(i),:)];
end

YS            =  repmat(ys,1,length(yt));
YT            =  repmat(yt,1,length(ys));
S             =  (YT==YS');
[HS,HT,H]     =  fea_trans(Xt',Xs',yt,ys,paras);
Dts           =  EuDist2(HT',HS');
Dp            =  S.*Dts;
[~,Ip]        =  max(Dp,[],2);
for i=1:length(yt)
    Xp         = [Xp; Xs(Ip(i),:)];%Similar sample
end
Dn            =  Dts-Dp;
[~,In]        =  min(Dn,[],2);
for i= 1:length(yt)
    Xn        = [Xn; Xs(In(i),:)];
end

Y              =     sparse(1:length(ys), double(ys), 1); 
Y              =     full(Y); 
L       = cl(H,paras);
D=zeros(paras.d,r);
F=0;

Bs = sign(2*rand(r,paras.ns )-1);
Bt = sign(2*rand(r,paras.nt )-1);

for iter=1:paras.max_iter
    for i=1:N
        xi=X1(i,:);
        xp=Xp(i,:);
        xn=Xn(i,:);
    
        if norm(W'*(xi-xp)','fro')-norm(W'*(xi-xn)','fro')+paras.m>=0
            omega=(1-exp(norm(W'*(xi-xn)','fro')-norm(W'*(xi-xp)','fro')-paras.m))^2;
            F=F+omega*norm(W'*(xi-xp)','fro')-omega*norm(W'*(xi-xn)','fro');
            D=D+omega*((xi-xn)'*(xi-xn)-(xi-xp)'*(xi-xp))*W;

        end
    end
    [W, ~]        = OptStiefelGBB(W, @W1_obj,setting,F,D,X,L,Xs,Xt,Bs,Bt,paras);
        %updata Bt 
    A=(paras.lambda2*(Bs*Bs')+paras.lambda3*eye(size(Bs*Bs')))\(paras.lambda2*Bs*Y);
    Bs            = sign((paras.lambda2*(A*A')+paras.theta1*eye(size(A*A')))\(paras.lambda2*A*Y'+paras.theta1 *W'*Xs'));
    Bt            = sign(W'*Xt');
end                

B_train           =    (Xs*W>0);
B_test            =    (test*W>0);
B_trn             =    compactbit(B_train);
B_tst             =    compactbit(B_test);	        

% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B_tst, B_trn);
[~, rank] = sort(Dhamm, 2, 'ascend');
clear B_tst B_trn;
choice = param.choice;
switch(choice)
    case 'evaluation_PR_MAP'
        clear train_data test_data;
        [recall, precision, ~] = recall_precision(WtrueTestTraining, Dhamm);
	[rec, pre]= recall_precision5(WtrueTestTraining, Dhamm, pos); % recall VS. the number of retrieved sample
        [mAP] = area_RP(recall, precision);
        retrieved_list = [];
    case 'evaluation_PR'
        clear train_data test_data;
        eva_info = eva_ranking(rank, trueRank, pos);
        rec = eva_info.recall;
        pre = eva_info.precision;
        recall = [];
        precision = [];
        mAP = [];
        retrieved_list = [];
    case 'visualization'
        num = param.numRetrieval;
        retrieved_list =  visualization(Dhamm, ID, num, train_data, test_data); 
        recall = [];
        precision = [];
        rec = [];
        pre = [];
        mAP = [];
end

end
