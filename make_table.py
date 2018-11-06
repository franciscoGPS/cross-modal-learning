import csv
def make_ids_table(file_name, array, openmode):
	with open(file_name+".csv",openmode) as my_csv:
		csvWriter = csv.writer(my_csv,delimiter=",")
		for line in array:
			excel_line =line.split(",")
			csvWriter.writerow(excel_line)