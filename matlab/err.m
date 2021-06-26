function E=err(x,xn,a,b)
syms s t;
tt=a:0.1:b;
E=vpa(norm(subs(subs(x-xn,s,t),t,tt),Inf),20);