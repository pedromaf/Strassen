import numpy as np
import sys

input_file_path = "in.txt"
output_file_path = "out.txt"

class TempMatrix:

	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.elements = initializeElements()

	def initializeElements():
		self.elements = np.zeros((self.rows, self.columns))	

class Matrix:

	def extraSize(self, square_size):
		
		#Check if the size isn't a power of 2.
		if(not (square_size > 0 and (square_size & (square_size - 1) == 0))):
			new_size = 1
			while(new_size < square_size):
				new_size = new_size << 1
			return (new_size - square_size)

	def __init__(self, base_rows, base_columns, square_size):

		self.base_rows = base_rows
		self.base_columns = base_columns
		self.square_size = self.extraSize(square_size)
		self.elements = self.initializeElements()

	def initializeElements(self):

		return np.zeros((int(self.square_size) * int(self.square_size))).reshape(self.square_size, self.square_size)


	def fillMatrix(self, input_file):

		print(self.base_rows)
		print(self.base_columns)
		print(range(self.base_rows))

		for i in range(self.base_rows):
			print("row " + str(i))
			row = [int(x) for x in input_file.readline().split()]
			for j in range(self.base_columns):
				print("column " + str(j))
				print("element " + str(row[j]))
				self.elements[i][j] = row[j]


def exit(input_file, output_file):

	if(input_file != None):
		input_file.close()
	if(output_file != None):
		output_file.close()
	sys.exit(0)		

def matrixAdd(matrix1, matrix1_index, matrix2, matrix2_index, output_matrix, output_matrix_index):

	matrix1_i, matrix1_j = matrix1_index
	matrix2_i, matrix2_j = matrix2_index
	output_matrix_i, output_matrix_j = output_matrix_index

	matrix1_current_i = matrix1_i[0]
	matrix1_current_j = matrix1_j[0]

	matrix2_current_i = matrix2_i[0]
	matrix2_current_j = matrix2_j[0]

	om_current_i = output_matrix_i[0]
	om_current_j = output_matrix_j[0]

	for _ in range(matrix1_i[1] - matrix1_i[0] + 1):
		for _ in range(matrix1_j[1] - matrix1_j[0] + 1):
			
			output_matrix.elements[om_current_i][om_current_j] = matrix1.elements[matrix1_current_i][matrix1_current_j] + matrix2.elements[matrix2_current_i][matrix2_current_j]
			
			matrix1_current_j +=1
			matrix2_current_j += 1
			om_current_j += 1

		matrix1_current_i += 1
		matrix2_current_i += 1
		om_current_i += 1

		matrix1_current_j = matrix1_j[0]
		matrix2_current_j = matrix2_j[0]
		om_current_j = output_matrix_j[0]

def matrixSubtract(matrix1, matrix1_index, matrix2, matrix2_index, output_matrix, output_matrix_index):

	matrix1_i, matrix1_j = matrix1_index
	matrix2_i, matrix2_j = matrix2_index
	output_matrix_i, output_matrix_j = output_matrix_index

	matrix1_current_i = matrix1_i[0]
	matrix1_current_j = matrix1_j[0]

	matrix2_current_i = matrix2_i[0]
	matrix2_current_j = matrix2_j[0]

	om_current_i = output_matrix_i[0]
	om_current_j = output_matrix_j[0]

	for _ in range(matrix1_i[1] - matrix1_i[0] + 1):
		for _ in range(matrix1_j[1] - matrix1_j[0] + 1):
			
			output_matrix.elements[om_current_i][om_current_j] = matrix1.elements[matrix1_current_i][matrix1_current_j] - matrix2.elements[matrix2_current_i][matrix2_current_j]
			
			matrix1_current_j +=1
			matrix2_current_j += 1
			om_current_j += 1

		matrix1_current_i += 1
		matrix2_current_i += 1
		om_current_i += 1

		matrix1_current_j = matrix1_j[0]
		matrix2_current_j = matrix2_j[0]
		om_current_j = output_matrix_j[0]

def strassen(matrix1, matrix1_index, matrix2, matrix2_index, output_matrix, output_matrix_index):

	matrix_size = (matrix1_index[0][1] - matrix1_index[0][0] + 1)

	if(matrix_size == 2):
		matrix1_i = matrix1_index[0][0]
		matrix1_j = matrix1_index[1][0]

		matrix2_i = matrix2_index[0][0]
		matrix2_j = matrix2_index[1][0]

		output_matrix_i = output_matrix_index[0][0]
		output_matrix_j = output_matrix_index[1][0]

		output_matrix[output_matrix_i][output_matrix_j] = (matrix1[matrix1_i][matrix1_j] * matrix2[matrix2_i][matrix2_j]) + (matrix1[matrix1_i][matrix1_j + 1] * matrix2[matrix2_i + 1][matrix2_j])
		output_matrix[output_matrix_i][output_matrix_j + 1] = (matrix1[matrix1_i][matrix1_j] * matrix2[matrix2_i][matrix2_j + 1]) + (matrix1[matrix1_i][matrix1_j + 1] * matrix2[matrix2_i + 1][matrix2_j + 1])
		output_matrix[output_matrix_i + 1][output_matrix_j] = (matrix1[matrix1_i + 1][matrix1_j] * matrix2[matrix2_i][matrix2_j]) + (matrix1[matrix1_i + 1][matrix1_j + 1] * matrix2[matrix2_i + 1][matrix2_j])
		output_matrix[output_matrix_i + 1][output_matrix_j + 1] = (matrix1[matrix1_i + 1][matrix1_j] * matrix2[matrix2_i][matrix2_j + 1]) + (matrix1[matrix1_i + 1][matrix1_j + 1] * matrix2[matrix2_i + 1][matrix2_j + 1])
	elif(matrix_size > 2):

		matrix1_i = matrix1_index[0]
		matrix1_j = matrix1_index[1]
		matrix1_mid_i = (matrix1_i[0] + matrix1_i[1]) / 2
		matrix1_mid_j = (matrix1_j[0] + matrix1_j[1]) / 2

		matrix2_i = matrix2_index[0]
		matrix2_j = matrix2_index[1]
		matrix2_mid_i = (matrix2_i[0] + matrix2_i[1]) / 2
		matrix2_mid_j = (matrix2_j[0] + matrix2_j[1]) / 2

		output_matrix_i = output_matrix_index[0]
		output_matrix_j = output_matrix_index[1]
		output_matrix_mid_i = (output_matrix_i[0] + output_matrix_i[1]) / 2
		output_matrix_mid_j = (output_matrix_j[0] + output_matrix_j[1]) / 2

		matrix1_a_index = ((matrix1_i[0], matrix1_mid_i), (matrix1_j[0], matrix1_mid_j))
		matrix1_b_index = ((matrix1_i[0], matrix1_mid_i), (matrix1_mid_j + 1, matrix1_j[1]))
		matrix1_c_index = ((matrix1_mid_i + 1, matrix1_i[1]), (matrix1_j[0], matrix1_mid_j))
		matrix1_d_index = ((matrix1_mid_i + 1, matrix1_i[1]), (matrix1_mid_j + 1, matrix1_j[1]))

		matrix2_a_index = ((matrix2_i[0], matrix2_mid_i), (matrix2_j[0], matrix2_mid_j))
		matrix2_b_index = ((matrix2_i[0], matrix2_mid_i), (matrix2_mid_j + 1, matrix2_j[1]))
		matrix2_c_index = ((matrix2_mid_i + 1, matrix2_i[1]), (matrix2_j[0], matrix2_mid_j))
		matrix2_d_index = ((matrix2_mid_i + 1, matrix2_i[1]), (matrix2_mid_j + 1, matrix2_j[1]))

		output_matrix_a_index = ((output_matrix_i[0], output_matrix_mid_i), (output_matrix_j[0], output_matrix_mid_j))
		output_matrix_b_index = ((output_matrix_i[0], output_matrix_mid_i), (output_matrix_mid_j + 1, output_matrix_j[1]))
		output_matrix_c_index = ((output_matrix_mid_i + 1, output_matrix_i[1]), (output_matrix_j[0], output_matrix_mid_j))
		output_matrix_d_index = ((output_matrix_mid_i + 1, output_matrix_i[1]), (output_matrix_mid_j + 1, output_matrix_j[1]))


		temp_1 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_2 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_3 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_4 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_5 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_6 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_7 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_8 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_9 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		temp_10 = TempMatrix((matrix_size / 2), (matrix_size / 2))

		temp_matrix_index = ((0, (matrix_size / 2) - 1), (0, (matrix_size / 2) - 1))

		matrixSubtract(matrix2, matrix2_b_index, matrix2, matrix2_d_index, temp_1, temp_matrix_index)
		matrixAdd(matrix1, matrix1_a_index, matrix1, matrix1_b_index, temp_2, temp_matrix_index)
		matrixAdd(matrix1, matrix1_c_index, matrix1, matrix1_d_index, temp_3, temp_matrix_index)
		matrixSubtract(matrix2, matrix2_c_index, matrix2, matrix2_a_index, temp_4, temp_matrix_index)
		matrixAdd(matrix1, matrix1_a_index, matrix1, matrix1_d_index, temp_5, temp_matrix_index)
		matrixAdd(matrix2, matrix2_a_index, matrix2, matrix2_d_index, temp_6, temp_matrix_index)
		matrixSubtract(matrix1, matrix1_b_index, matrix1, matrix1_d_index, temp_7, temp_matrix_index)
		matrixAdd(matrix2, matrix2_c_index, matrix2, matrix2_d_index, temp_8, temp_matrix_index)
		matrixSubtract(matrix1, matrix1_a_index, matrix1, matrix1_c_index, temp_9, temp_matrix_index)
		matrixAdd(matrix2, matrix2_a_index, matrix2, matrix2_b_index, temp_10, temp_matrix_index)

		aux_1 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_2 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_3 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_4 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_5 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_6 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		aux_7 = TempMatrix((matrix_size / 2), (matrix_size / 2))

		aux_1 = strassen(matrix1, matrix1_a_index, temp_1, temp_matrix_index, aux_1, temp_matrix_index)
		aux_2 = strassen(temp_2, temp_matrix_index, matrix2, matrix2_d_index, aux_2, temp_matrix_index)
		aux_3 = strassen(temp_3, temp_matrix_index, matrix2, matrix2_a_index, aux_3, temp_matrix_index)
		aux_4 = strassen(matrix1, matrix1_d_index, temp_4, temp_matrix_index, aux_4, temp_matrix_index)
		aux_5 = strassen(temp_5, temp_matrix_index, temp_6, temp_matrix_index, aux_5, temp_matrix_index)
		aux_6 = strassen(temp_7, temp_matrix_index, temp_8, temp_matrix_index, aux_6, temp_matrix_index)
		aux_7 = strassen(temp_9, temp_matrix_index, temp_10, temp_matrix_index, aux_7, temp_matrix_index)

		output_aux_1 = TempMatrix((matrix_size / 2), (matrix_size / 2))
		output_aux_2 = TempMatrix((matrix_size / 2), (matrix_size / 2))

		matrixAdd(aux_5, temp_matrix_index, aux_4, temp_matrix_index, output_aux_1, temp_matrix_index)
		matrixAdd(aux_2, temp_matrix_index, aux_6, temp_matrix_index, output_aux_2, temp_matrix_index)
		matrixSubtract(output_aux_1, temp_matrix_index, output_aux_2, temp_matrix_index, output_matrix, output_matrix_a_index)

		matrixAdd(aux_1, temp_matrix_index, aux_2, temp_matrix_index, output_matrix, output_matrix_b_index)

		matrixAdd(aux_3, temp_matrix_index, aux_4, temp_matrix_index, output_matrix, output_matrix_c_index)

		matrixAdd(aux_5, temp_matrix_index, aux_1, temp_matrix_index, output_aux_1, temp_matrix_index)
		matrixSubtract(aux_3, temp_matrix_index, aux_7, temp_matrix_index, output_aux_2, temp_matrix_index)
		matrixSubtract(output_aux_1, temp_matrix_index, output_aux_2, temp_matrix_index, output_matrix, output_matrix_d_index)

	return output_matrix


def main():
	
	try:
		input_file = open(input_file_path)
	except IOError as exception:
		print("Error opening the input file!")
		exit(None, None)

	try:
		output_file = open(output_file_path, "w")
	except IOError as exception:
		print("Error creating/opening the output file!")
		exit(input_file, None)

	try:
		matrix1_rows, matrix1_columns, matrix2_rows, matrix2_columns = [int(x) for x in input_file.readline().split()]
		
		if(matrix1_columns != matrix2_rows):
			print("These matrices can not be multiplied!")
			exit(input_file, output_file)

		square_size = int(max(matrix1_rows, max(matrix2_columns, matrix1_columns)))
		matrix1 = Matrix(matrix1_rows, matrix1_columns, square_size)
		matrix2 = Matrix(matrix2_rows, matrix2_columns, square_size)
	except IOError as exception:
		print("Error reading the input file!")
		exit(input_file, output_file)

	output_matrix = Matrix(matrix1_rows, matrix2_columns, square_size)

	matrix1.fillMatrix(input_file)
	matrix2.fillMatrix(input_file)

	matrix1_index = ((0, matrix1.square_size),(0, matrix1.square_size))
	matrix2_index = ((0, matrix2.square_size),(0, matrix2.square_size))
	matrix1_index = ((0, output_matrix.square_size),(0, output_matrix.square_size))

	output_matrix = strassen(matrix1, matrix1_index, matrix2, output_matrix)

	output_file.write("16113134\n")
	output_file.write(str(output_matrix.base_rows) + " " + str(output_matrix.base_columns) + "\n")
	for i in range(output_matrix.base_rows):
		for j in range(output_matrix.base_columns):
			output_file.write(str(output_matrix.elements[i][j]) + " ")
		output_file.write("\n")

		
if __name__ == '__main__':
	main()