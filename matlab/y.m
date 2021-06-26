function y=y(t)
syms s t
lamuda=5;b=1;
x=exp(-t).*cos(t);
z=exp(s*t)*x;
f1=int(z,t,0,b);
y=lamuda*x-f1;
y=subs(y,t,s);
