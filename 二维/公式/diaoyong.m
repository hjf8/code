S=[1 2 3 4;0 5 6 7;0 0 8 9;0 0 0 10];
T=[10 9 8 7;0 2 6 5;0 0 1 4;0 0 0 3];
b=[8 7 5 3]';
lamuda=4;
[x]=Block(S,T,lamuda,b)