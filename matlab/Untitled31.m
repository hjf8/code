 A=[];B=[];Aeq=[];Beq=[];xm=[-10;-10],xM=[10;10];
x0=[5;5];ff=optimset;ff.Tolx=1e-10;ff.TolFUN=1e-20;
x=fmincon(@c3exmobj,x0,A,B,Aeq,Beq,xm,xM,@c3exmcon,ff)

i=1;x=x0;
while(1)
    [x,a,b]=fmincon(@c3exmobj,x,A,B,Aeq,Beq,xm,xM,@c3exmcon,ff);
    if b>0,break;end
    i=i+1;
end