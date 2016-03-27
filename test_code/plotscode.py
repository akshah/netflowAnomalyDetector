
    fig=plt.figure(2,figsize=(20, 16))
	plt.suptitle('NEWS Flow Forensics: Netflow from a CA Customer',fontsize=14)
	total_subplot=6
	subnum=1

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(1,5):
		plt.plot(standardize(data[:,c]),label=names[c])
	
	plt.ylabel('Entropy')

	plt.legend(loc='lower right')
	
    plt.tick_params(\
	    axis='x',          # changes apply to the x-axis
	     which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off'
	  ) # labels along the bottom edge are off  

	
	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(5,7):
		plt.plot(standardize(data[:,c]),label=names[c])
	plt.legend(loc='lower right')


	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(7,8):
	    plt.plot(standardize(data[:,c]),label=names[c])
	plt.ylabel('Avg Packet Size')
	plt.legend(loc='lower right')

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(8,10):
	    plt.plot(standardize(data[:,c]),label=names[c])
	plt.legend(loc='lower right')
	
	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	plt.plot(standardize(data[:,10]),label=names[10])
	plt.legend(loc='lower right')
	plt.ylabel('Numflows/min')
	
	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	plt.plot(anomaly_score,'-',label='Anomaly Score')
	for mark in markers_on:
		val=anomaly_score[mark]
		plt.plot(mark,val,'ro')
	plt.ylim(0.1,15)
	plt.ylabel('Score')

	plt.legend(loc='lower right')'''

	#xticks=['','3am','6pm','9pm','']
	#plt.xticks(np.arange(1,nrow,nrow/24*6),xticks)
