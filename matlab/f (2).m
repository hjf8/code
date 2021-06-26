function y=y(t)
syms s t
lamuda=5;b=1;
x1=exp(-t).*cos(t);
x2=exp(-s).*cos(s);
z=exp(s*t)*x2;
f1=int(z,s,0,b);
y=lamuda*x1-f1;
disp(y)
