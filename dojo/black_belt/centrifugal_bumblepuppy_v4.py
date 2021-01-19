code = compile(
    """
#                         WoofWoofWoof
#                     Woof            Woof
#                Woof                      Woof
#              Woof                          Woof
#             Woof  Centrifugal Bumble-puppy  Woof
#              Woof                          Woof
#                Woof                      Woof
#                     Woof            Woof
#                         WoofWoofWoof

import random

number_of_predictors = 60 #yes, this really has 60 predictors.
number_of_metapredictors = 4 #actually, I lied! This has 240 predictors.


if not input:
	limits = [50,20,6]
	beat={'R':'P','P':'S','S':'R'}
	urmoves=""
	mymoves=""
	DNAmoves=""
	outputs=[random.choice("RPS")]*number_of_metapredictors
	predictorscore1=[3]*number_of_predictors
	predictorscore2=[3]*number_of_predictors
	predictorscore3=[3]*number_of_predictors
	predictorscore4=[3]*number_of_predictors
	nuclease={'RP':'a','PS':'b','SR':'c','PR':'d','SP':'e','RS':'f','RR':'g','PP':'h','SS':'i'}
	length=0
	predictors=[random.choice("RPS")]*number_of_predictors
	metapredictors=[random.choice("RPS")]*number_of_metapredictors
	metapredictorscore=[3]*number_of_metapredictors
else:

	for i in range(number_of_predictors):
		#metapredictor 1
		predictorscore1[i]*=0.8
		predictorscore1[i]+=(input==predictors[i])*3
		predictorscore1[i]-=(input==beat[beat[predictors[i]]])*3
		#metapredictor 2: beat metapredictor 1 (probably contains a bug)
		predictorscore2[i]*=0.8
		predictorscore2[i]+=(output==predictors[i])*3
		predictorscore2[i]-=(output==beat[beat[predictors[i]]])*3
		#metapredictor 3
		predictorscore3[i]+=(input==predictors[i])*3
		if input==beat[beat[predictors[i]]]:
			predictorscore3[i]=0
		#metapredictor 4: beat metapredictor 3 (probably contains a bug)
		predictorscore4[i]+=(output==predictors[i])*3
		if output==beat[beat[predictors[i]]]:
			predictorscore4[i]=0
			
	for i in range(number_of_metapredictors):
		metapredictorscore[i]*=0.96
		metapredictorscore[i]+=(input==metapredictors[i])*3
		metapredictorscore[i]-=(input==beat[beat[metapredictors[i]]])*3
		
	
	#Predictors 1-18: History matching
	urmoves+=input		
	mymoves+=output
	DNAmoves+=nuclease[input+output]
	length+=1
	
	for z in range(3):
		limit = min([length,limits[z]])
		j=limit
		while j>=1 and not DNAmoves[length-j:length] in DNAmoves[0:length-1]:
			j-=1
		if j>=1:
			i = DNAmoves.rfind(DNAmoves[length-j:length],0,length-1) 
			predictors[0+6*z] = urmoves[j+i] 
			predictors[1+6*z] = beat[mymoves[j+i]] 
		j=limit			
		while j>=1 and not urmoves[length-j:length] in urmoves[0:length-1]:
			j-=1
		if j>=1:
			i = urmoves.rfind(urmoves[length-j:length],0,length-1) 
			predictors[2+6*z] = urmoves[j+i] 
			predictors[3+6*z] = beat[mymoves[j+i]] 
		j=limit
		while j>=1 and not mymoves[length-j:length] in mymoves[0:length-1]:
			j-=1
		if j>=1:
			i = mymoves.rfind(mymoves[length-j:length],0,length-1) 
			predictors[4+6*z] = urmoves[j+i] 
			predictors[5+6*z] = beat[mymoves[j+i]]
	#Predictor 19,20: RNA Polymerase		
	L=len(mymoves)
	i=DNAmoves.rfind(DNAmoves[L-j:L-1],0,L-2)
	while i==-1:
		j-=1
		i=DNAmoves.rfind(DNAmoves[L-j:L-1],0,L-2)
		if j<2:
			break
	if i==-1 or j+i>=L:
		predictors[18]=predictors[19]=random.choice("RPS")
	else:
		predictors[18]=beat[mymoves[j+i]]
		predictors[19]=urmoves[j+i]

	#Predictors 21-60: rotations of Predictors 1:20
	for i in range(20,60):
		predictors[i]=beat[beat[predictors[i-20]]] #Trying to second guess me?
	
	metapredictors[0]=predictors[predictorscore1.index(max(predictorscore1))]
	metapredictors[1]=beat[predictors[predictorscore2.index(max(predictorscore2))]]
	metapredictors[2]=predictors[predictorscore3.index(max(predictorscore3))]
	metapredictors[3]=beat[predictors[predictorscore4.index(max(predictorscore4))]]
	
	#compare predictors
output = beat[metapredictors[metapredictorscore.index(max(metapredictorscore))]]
if max(metapredictorscore)<0:
	output = beat[random.choice(urmoves)]
""", '<string>', 'exec')
gg = {}


def run(observation, configuration):
    global gg
    global code
    inp = ''
    try:
        inp = 'RPS'[observation.lastOpponentAction]
    except:
        pass
    gg['input'] = inp
    exec(code, gg)
    return {'R': 0, 'P': 1, 'S': 2}[gg['output']]
