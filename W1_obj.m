function [ Ft, Gt] = W1_obj(W,F,D,X,L,Xs,Xt,Bs,Bt,paras)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Ft       = F+paras.theta1 *norm(Bs-W'*Xs','fro')^2+paras.theta2*norm(Bt-W'*Xt','fro')^2+paras.lambda1*trace(W'*X'*L*X*W);
Gt       = 2*D-2*paras.theta1 *Xs'*Bs'+2*paras.theta1 *(Xs'*Xs)*W-2*paras.theta2 *Xt'*Bt'+2*paras.theta2 *(Xt'*Xt)*W+2*paras.lambda1*X'*L*X*W;
end