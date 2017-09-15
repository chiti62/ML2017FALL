import sys

f = open(sys.argv[1],"r")
contents = f.read()
words = contents.split()
f.close()

ans = {}
order = []

for word in words:
	for key in ans:
		if key == word:
			ans[key] += 1
			break
	else:
		order.append(word)
		ans[word] = 1

f = open("Q1.txt", "w")

for i in range(len(order)):
	#print (order[i] + " " + str(i) + " " + str(ans[key]) + '\n')
	f.write(order[i] + " " + str(i) + " " + str(ans[order[i]]) + '\n')

f.close()