function [c,s]=givens(a,b)
if b==0
    c=1;s=0;
else
    if abs(b)>abs(a)
        ta=-a/b;s=1/(sqrt(1+ta^2));c=s*ta;
    else
        ta=-b/a;c=1/(sqrt(1+ta^2));s=c*ta;
    end
end


