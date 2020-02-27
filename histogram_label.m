function [h] = histogram_label(l,classnum,k)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
h = zeros(classnum,1);
for i =1:classnum
   num   = length(find(i==l));
   h(i)  = num; 
end
h = h/k;
end

