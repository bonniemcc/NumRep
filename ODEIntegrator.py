# Guiding program

def main():

    type = (input( "Enter type of funcion (exp, sin, poly) "))
    nsteps = int( input( "Enter number of steps ") )
    delta = float( input( "Enter step size "))


    print ' Running ODE integration for ', type, ' with nsteps = ',nsteps,'  and dt = ',delta

    #Create ODE
    if( type == 'poly' ):
        ode = MyODEPolynomial([ 1., 1., -3., 1. ] )
#        ode = MyODEPolynomial([ 10.2, -7.42, -2.1, 1. ] )
    elif( type == 'exp' ):
        ode = MyODEExponential(1. )
    elif( type == 'sin' ):
        ode = MyODESinusoid( 1. )
    else:
	print ' No functional form specified: ', type
        quit()

    #Create Euler step machine
    eulerStep = StepEuler( )
    eulerEngine = Engine( ode, eulerStep, ' Exponential with Euler integration ')

    #Create RK0 step machine
    rk0Step = StepRK0( )
    rk0Engine = Engine( ode, rk0Step, ' Exponential with RK-0 integration ')

    #Create RK step machine
    rkStep = StepRK( )
    rkEngine = Engine( ode, rkStep, ' Exponential with RK integration ')

    #run them
    resultSetEuler = eulerEngine.go(nsteps, delta)
    resultSetRK0 = rk0Engine.go(nsteps, delta)
    resultSetRK = rkEngine.go(nsteps, delta)

    yvaluesEuler = resultSetEuler.yvalues()
    yvaluesRK0 = resultSetRK0.yvalues()
    yvaluesRK = resultSetRK.yvalues()
    tvalues = resultSetRK.tvalues()

    #print yvalues
    #yexact = [ mt.exp( -t ) for t in tvalues ]
    yexact = [ ode.exactSolution( pair[1] ) for pair in resultSetEuler.getAll() ]

    plot(tvalues,yvaluesEuler,'g',label='Euler Results')
    plot(tvalues,yvaluesRK0,'y-',label='RK Results')
    plot(tvalues,yvaluesRK,'b--',label='RK Results')
    plot(tvalues,yexact,'ro',label='Exact Results')
    show()
    #pl.hist(y,color='b',bins=nsteps,normed='True',label='Euler Results')
#    pl.hist(GaussSamples,color='g',bins=50,normed='True',label='Normal distribution')
#    pl.hist(LnSamples,color='y',bins=50,normed='True',label='Lognormal dist.', alpha=0.7)
#    pl.xlim([-20.,20.])
#    pl.xlabel(r'$x$')
#    pl.ylabel(r'$p(x)$')
#    leg = pl.legend(loc=1, ncol=1, prop=FontProperties(size='8'))
#    leg.draw_frame(False)
#    pl.savefig("NumericalRecipes_PythonRandomNumGenerators.pdf")




main()
