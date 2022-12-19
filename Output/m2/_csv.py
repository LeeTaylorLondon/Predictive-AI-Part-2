import csv

with open("2.txt", 'r') as f:
    text = f.readlines()

del text[0]
text.pop()

# Open a file for writing
with open('2.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Loop through each line of the text
    for line in text:
        # Split the line by ': '
        p = line.split(': ')
        scors = p[8][1:-2].split(',')
        wpart = [p[2].split(',')[0], p[3].split(',')[0], p[4].split(',')[0],
                 p[5].split(',')[0], p[6].split(',')[0], p[7].split(',')[0],
                 scors[0], scors[1]]
        # Write the description and scores to the .csv file as separate cells
        writer.writerow(wpart)

