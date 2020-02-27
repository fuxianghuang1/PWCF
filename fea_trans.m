function [HS,HT,H] = fea_trans(DT,DS,YT,YS,paras)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
YT                         = YT'; % row label vector
YS                         = YS'; % row label vector
k                          = paras.k;
ns                         = paras.ns;
nt                         = paras.nt;
HS                         = [];  
HT                         = [];
SS                         = EuDist2(DS',DS');
SS(logical(eye(size(SS)))) = 10000;
ST                         = EuDist2(DT',DT');
ST(logical(eye(size(ST)))) = 10000;

clsnum1                    = length(unique(YS));
clsnum2                    = length(unique(YT));
if clsnum1==clsnum2
    clsnum = clsnum1;
else
    fprintf('......%s start ......\n\n', 'inequal class number for source and target domains');
    clsnum = clsnum1;
end
for i=1:size(SS,1)
 index        = zeros(1,ns);
 [~,ind]      = sort(SS(i,:),'ascend');
 ind          = ind(1:k);
 index(ind)   = 1;
 label_appear = YS.*index;
 hs           = histogram_label(label_appear,clsnum,k);
 HS           = [HS,hs];
end
clear SS;clear ind;clear index;clear hs;clear label_appear;clear YS;
for j =1:size(ST,1)
 index        = zeros(1,nt);
 [~,ind] = sort(ST(j,:),'ascend');
 ind          = ind(1:k);
 index(ind)   = 1;
 label_appear = YT.*index;
 ht           = histogram_label(label_appear,clsnum,k);
 HT           = [HT,ht];   
end
clear ST;clear ind;clear index;clear ht;clear label_appear;clear YT;
H = [HS,HT];
end

