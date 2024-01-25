import random
from copy import deepcopy
from typing import Tuple

import numpy as np
from scipy import linalg

__all__ = ['Ransac', 'PositionRansac', 'GeneticRansac', 'NewGeneticRansac', 'GeneticTransform']


class Ransac:

    def __init__(self, data1: np.ndarray, data2: np.ndarray, max_iter_times=1000):
        random.seed(19270817)
        # print(repr(data1), repr(data2))
        self.data1 = data1
        self.data2 = data2
        if(self.data1.shape != self.data2.shape):
            raise ValueError("Argument shape not equal")

        self.points_length = self.data1.shape[0]
        self.good_points = 0
        self.max_iter_times = max_iter_times

        self.mask = np.zeros(self.points_length, dtype=np.bool)

    def random_calculate(self, max_try_times=1000)-> np.ndarray:
        if self.points_length - np.count_nonzero(self.mask) < 4:
            return False

        rand_point = random.sample(range(self.points_length), 4)
        M = self.get_perspective_transform(self.data1[rand_point], self.data2[rand_point])
        return M

    @staticmethod
    def get_perspective_transform(src: np.ndarray, dst: np.ndarray)-> np.ndarray:
        X = np.array((8, 1), np.float)
        A = np.zeros((8, 8), np.float)
        B = np.zeros((8), np.float)

        for i in range(4):
            A[i][0] = A[i + 4][3] = src[i][0]
            A[i][1] = A[i + 4][4] = src[i][1]
            A[i][2] = A[i + 4][5] = 1
            A[i][3] = A[i][4] = A[i][5] = A[i + 4][0] = A[i + 4][1] = A[i + 4][2] = 0
            A[i][6] = -src[i][0] * dst[i][0]
            A[i][7] = -src[i][1] * dst[i][0]
            A[i + 4][6] = -src[i][0] * dst[i][1]
            A[i + 4][7] = -src[i][1] * dst[i][1]
            B[i] = dst[i][0]
            B[i + 4] = dst[i][1]
        try:
            X = linalg.solve(A, B).copy()
        except Exception as e:
            return False
        else:
            X.resize((3, 3), refcheck=False)
            X[2][2] = 1
        return X

    @staticmethod
    def perspective_transform(points: np.ndarray, M: np.ndarray)->np.ndarray:
        array_i = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype))).T
        x = np.dot(M, array_i)
        result = np.vstack((x[0] / x[2], x[1] / x[2])).T
        return result
    
    @classmethod
    def get_good_points(cls, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3)-> int:
        transformed = cls.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold
        return np.sum(good)

    @staticmethod
    def get_itereration_time(proportion: float, p=0.995, points=4)->int:
        proportion = max(min(proportion, 1), 0)
        p = max(min(p, 1), 0)
        # print(p,proportion)
        k = np.log(1 - p) / np.log(1 - np.power(proportion, 4))
        return int(k)

    def run(self, distance=3)->np.ndarray:
        iter_times = 0
        best_M = None
        while iter_times < self.max_iter_times:
            M = self.random_calculate()
            if M is False:
                break
            good_nums = self.get_good_points(self.data1, self.data2, M, distance)
            if good_nums > self.good_points:
                self.good_points = good_nums
                best_M = M
                self.max_iter_times = min(self.max_iter_times,
                                          self.get_itereration_time(good_nums / self.points_length))
            iter_times += 1


        return best_M


class PositionRansac(Ransac):

    def __init__(self, data1: np.ndarray, data2: np.ndarray):
        raise NotImplementedError
        super().__init__(data1, data2)

    def random_calculate(self, index=None)-> np.ndarray:
        if index is not None:
            rand_point = random.sample(range(self.points_length), 3)
            while index in rand_point:
                rand_point = random.sample(range(self.points_length), 3)
            rand_point.append(index)
        else:
            rand_point = random.sample(range(self.points_length), 4)

        M = self.get_perspective_transform(self.data1[rand_point], self.data2[rand_point])
        return M

    def run_ransac(self, index=None) -> np.ndarray:
        max_iter_times = self.max_iter_times
        iter_times = 0
        best_M = None
        # random.seed(1)
        while iter_times < max_iter_times:
            M = self.random_calculate(index)
            if M is False:
                return best_M
            good_nums = self.get_good_points(self.data1, self.data2, M)
            if good_nums > self.good_points:
                self.good_points = good_nums
                best_M = M
                max_iter_times = min(max_iter_times,
                                     self.get_itereration_time(good_nums / self.points_length))
            iter_times += 1

        return best_M

    def get_farthest_point(self, M: np.ndarray) -> int:
        transformed = self.perspective_transform(self.data1, M)
        dis = np.sum((transformed - self.data2) * (transformed - self.data2), axis=1)
        farthest = np.argmax(dis)
        return farthest


class GeneticRansac(Ransac):

    class Individual:
        def __init__(self, dna, value=0):
            self.dna = dna
            self.value = value

        def __lt__(self, other: 'Individual'):
            return self.value < other.value

        def __repr__(self):
            return "dna: <{}>, value: {}".format(self.dna, self.value)

        __str__ = __repr__

    SAMPLE = 30
    MUTATION_RATE = 0.1

    def __init__(self, data1: np.ndarray, data2: np.ndarray):
        super().__init__(data1, data2)
        if self.points_length > 7:
            self.population = []
            for i in range(self.SAMPLE):
                indv = self.Individual(np.random.choice(range(self.points_length), 4, replace=0))
                self.population.append(indv)

    def run(self):
        if self.points_length < 7:
            return super().run()

        for i in range(40):
            for i in self.population:
                M = self.get_perspective_transform(self.data1[i.dna], self.data2[i.dna])
                if M is not False:
                    good = self.get_good_points(self.data1, self.data2, M)
                    i.value = good
                else:
                    i.value = 0
            self.population = sorted(self.population, reverse=True)

            all_value = sum([x.value for x in self.population])
            prop = [x.value / all_value for x in self.population]

      
            self.population = np.random.choice(self.population, size=self.SAMPLE // 2, replace=False, p=prop)

            children = []
            while len(children) + len(self.population) <= self.SAMPLE:
                mother = random.choice(self.population)
                father = random.choice(self.population)
                corss_index = random.choices(range(4), k=2)
                child = deepcopy(mother)
                child.dna[corss_index] = father.dna[corss_index]
                if random.random() < self.MUTATION_RATE:
                    rand_index = random.choice(range(4))
                    rand_point = random.choice(range(self.points_length))
                    while child.dna[rand_index] == rand_point:
                        rand_point = random.choice(range(self.points_length))
                    child.dna[rand_index] = rand_point

                children.append(child)

            self.population = np.concatenate((self.population, children))
        M = self.get_perspective_transform(self.data1[self.population[0].dna], self.data2[self.population[0].dna])

        i = 1
        while M is False:
            M = self.get_perspective_transform(self.data1[self.population[i].dna], self.data2[self.population[i].dna])
            i += 1
            if i > self.SAMPLE:
                raise Exception("GA error!!")
        self.good_points = self.get_good_points(self.data1, self.data2, M)
        return M


class NewGeneticRansac(Ransac):

    class Individual:
        def __init__(self, dna: np.ndarray, value=0):
            self.dna = dna
            self.value = value

        def __lt__(self, other: 'Individual'):
            return self.value < other.value

        def __repr__(self):
            return "dna: <{}>, value: {}".format(self.dna, self.value)

        __str__ = __repr__

    SAMPLE = 30
    GENERATION = 20
    MUTATION_RATE = 0.1
    MUTATION_COUNT = 1

    def __init__(self, data1: np.ndarray, data2: np.ndarray):
        self.data1 = data1
        self.data2 = data2
        if(self.data1.shape != self.data2.shape):
            raise ValueError("Argument shape not equal")

        self.points_length = self.data1.shape[0]

        self.population = []
        for i in range(self.SAMPLE):
            ransac = Ransac(data1, data2)
            M = ransac.run()
            indv = self.Individual(M, ransac.good_points)
            self.population.append(indv)

    def run(self):

        for i in range(self.GENERATION):

            self.population = sorted(self.population, reverse=True)

            self.population = self.population[:self.SAMPLE // 2]


            children = []
            while len(children) + len(self.population) <= self.SAMPLE:
                mother = random.choice(self.population)
                father = random.choice(self.population)
                child = self.Individual((mother.dna + father.dna) / 2)

                if random.random() < self.MUTATION_RATE:
                    rand_index = np.random.choice(range(8), self.MUTATION_COUNT, replace=0)
                    gene = child.dna.take(rand_index)
                    gene *= np.random.chisquare(10, rand_index.size)
                    child.dna.put(rand_index, gene)
                child.value = self.get_good_points(self.data1, self.data2, child.dna)
                children.append(child)

            self.population = np.concatenate((self.population, children))

        # 获取最优点
        self.population = sorted(self.population, reverse=True)
        if self.population[0].value > 0:
            self.good_points = self.population[0].value
            return self.population[0].dna
        else:
            raise RuntimeError("Cannot get transfrom matrix")


class GeneticTransform(Ransac):

    class Individual:
        def __init__(self, dna: np.ndarray, value=0):
            self.dna = dna
            self.value = value

        def __lt__(self, other: 'Individual'):
            return self.value < other.value

        def __repr__(self):
            return "dna: <{}>, value: {}".format(self.dna, self.value)

        __str__ = __repr__

    SAMPLE = 40
    GROUP_SIZE = 8
    GENERATION = 20
    MUTATION_RATE = 0.2
    MUTATION_COUNT = 1

    def __init__(self, data1: np.ndarray, data2: np.ndarray):
        random.seed(19580829)
        np.random.seed(19580829)

        self.data1 = data1
        self.data2 = data2
        if(self.data1.shape != self.data2.shape):
            raise ValueError("Argument shape not equal")

        self.points_length = self.data1.shape[0]
        if self.points_length < 4:
            raise RuntimeError("Not enough points to calculate")
        self.GROUP_SIZE = min(self.GROUP_SIZE, max(self.points_length // 2, 4))

        self.population = []
        try_times = 0

        while len(self.population) < self.SAMPLE:
            try_times += 1
            if try_times > self.SAMPLE * 4:
                if len(self.population) > 1:
                    break
                else:
                    raise RuntimeError("Cannot find enough points")
            choice = np.random.choice(range(self.points_length), self.GROUP_SIZE, replace=0)
            M = self.get_lss_matrix(self.data1[choice], self.data2[choice])
            good_points, distance = self.get_value(self.data1, self.data2, M)
            if good_points < 4:
                continue
            indv = self.Individual(M, self.get_judgement(self.data1, self.data2, M))
            self.population.append(indv)

    @classmethod
    def get_judgement(cls, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3) -> float:
        transformed = cls.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold
        good_points = np.sum(good)
        if good_points == 0:
            return -np.sum(dis)

        distance = np.sum(dis[good])
        return good_points - distance / good_points

    @staticmethod
    def get_lss_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data_length = data1.shape[0]
        X = np.array((8, 1), np.float)
        A = np.zeros((2 * data_length, 8), np.float)
        B = np.zeros((2 * data_length), np.float)

        for i in range(data_length):
            A[2 * i][0] = A[2 * i + 1][3] = data1[i][0]
            A[2 * i][1] = A[2 * i + 1][4] = data1[i][1]
            A[2 * i][2] = A[2 * i + 1][5] = 1
            A[2 * i][3] = A[2 * i][4] = A[2 * i][5] = A[2 * i + 1][0] = A[2 * i + 1][1] = A[2 * i + 1][2] = 0
            A[2 * i][6] = -data1[i][0] * data2[i][0]
            A[2 * i][7] = -data1[i][1] * data2[i][0]
            A[2 * i + 1][6] = -data1[i][0] * data2[i][1]
            A[2 * i + 1][7] = -data1[i][1] * data2[i][1]
            B[2 * i] = data2[i][0]
            B[2 * i + 1] = data2[i][1]

        try:
            dot = np.dot
            AT = A.T
            X = dot(dot(linalg.inv(np.dot(AT, A)), AT), B)

        except Exception as e:
            print("Error when calcuating M: ", e)
            return False
        else:
            X.resize((3, 3), refcheck=False)
            X[2][2] = 1
        return X

    @classmethod
    def get_value(cls, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3)-> Tuple[int, float]:

        transformed = cls.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold
        error = np.sum(dis[good])
        return np.sum(good), error

    def run(self):

        data = []
        for i in range(self.GENERATION):

            self.population = sorted(self.population, reverse=True)
            data.append([j.value for j in self.population[:min(4, len(self.population))]])
            self.population = self.population[:len(self.population) // 2]
          
            for j in range(min(12, 4 * len(self.population))):
                new = deepcopy(self.population[j // 3])
                rand_index = np.random.choice(range(8), self.MUTATION_COUNT, replace=0)
                gene = new.dna.take(rand_index)
                gene *= np.random.normal(1, .1, rand_index.size)
                new.dna.put(rand_index, gene)
                new.value = self.get_judgement(self.data1, self.data2, new.dna)
                self.population.append(new)

            children = []
            while len(children) + len(self.population) <= self.SAMPLE:
                mother = random.choice(self.population)
                father = random.choice(self.population)
                corss_index = random.choices(range(8), k=4)
                child = deepcopy(mother)
                child.dna.put([corss_index], father.dna.take([corss_index]))
          
                if random.random() < self.MUTATION_RATE:
                    rand_index = np.random.choice(range(8), self.MUTATION_COUNT, replace=0)
                    gene = child.dna.take(rand_index)
                    gene *= np.random.chisquare(2, rand_index.size)
                    child.dna.put(rand_index, gene)
                child.value = self.get_judgement(self.data1, self.data2, child.dna)
                children.append(child)

            self.population = np.concatenate((self.population, children))

        self.log(np.array(data).T)
        self.population = sorted(self.population, reverse=True)
        if self.population[0].value > 3:
            self.good_points = self.population[0].value
            return self.population[0].dna
        else:
            raise RuntimeError("Cannot get transfrom matrix")

class GeneticTransformWithPosition(Ransac):

    class Individual:
        def __init__(self, dna: np.ndarray, value=0):
            self.dna = dna
            self.value = value

        def __lt__(self, other: 'Individual'):
            return self.value < other.value

        def __repr__(self):
            return "dna: <{}>, value: {}".format(self.dna, self.value)

        __str__ = __repr__

    SAMPLE = 40
    GROUP_SIZE = 8
    GENERATION = 20
    MUTATION_RATE = 0.2
    MUTATION_COUNT = 1

    def __init__(self, data1: np.ndarray, data2: np.ndarray):
        random.seed(19580829)
        np.random.seed(19580829)

        self.data1 = data1
        self.data2 = data2
        if(self.data1.shape != self.data2.shape):
            raise ValueError("Argument shape not equal")

        self.points_length = self.data1.shape[0]
        if self.points_length < 4:
            raise RuntimeError("Not enough points to calculate")
       
        centroid = np.sum(data1, axis=0) / self.points_length
        self.distance = np.sum(np.square(data1 - centroid)) / self.points_length

        self.GROUP_SIZE = min(self.GROUP_SIZE, max(self.points_length // 2, 4))

        self.population = []
        try_times = 0

        while len(self.population) < self.SAMPLE:
            try_times += 1
            if try_times > self.SAMPLE * 4:
                if len(self.population) > 1:
                    break
                else:
                    raise RuntimeError("Cannot find enough points")
            choice = np.random.choice(range(self.points_length), self.GROUP_SIZE, replace=0)
            M = self.get_lss_matrix(self.data1[choice], self.data2[choice])
            good_points, distance = self.get_value(self.data1, self.data2, M)
            if good_points < 4:
                continue
            indv = self.Individual(M, self.get_judgement(self.data1, self.data2, M))
            self.population.append(indv)

    def get_judgement(self, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3) -> float:
        transformed = self.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold
        good_points = np.sum(good)

        if good_points == 0:
            return -np.sum(dis)

        centroid = np.sum(points1[good], axis=0) / good_points
        distance = np.sum(np.square(points1[good] - centroid))
        stand_error = np.tanh(((distance / good_points) - self.distance) / self.distance) + 1 
        interier_error = np.sum(dis[good]) / good_points

        return good_points + stand_error - interier_error

    @staticmethod
    def get_lss_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data_length = data1.shape[0]
        X = np.array((8, 1), np.float)
        A = np.zeros((2 * data_length, 8), np.float)
        B = np.zeros((2 * data_length), np.float)

        for i in range(data_length):
            A[2 * i][0] = A[2 * i + 1][3] = data1[i][0]
            A[2 * i][1] = A[2 * i + 1][4] = data1[i][1]
            A[2 * i][2] = A[2 * i + 1][5] = 1
            A[2 * i][3] = A[2 * i][4] = A[2 * i][5] = A[2 * i + 1][0] = A[2 * i + 1][1] = A[2 * i + 1][2] = 0
            A[2 * i][6] = -data1[i][0] * data2[i][0]
            A[2 * i][7] = -data1[i][1] * data2[i][0]
            A[2 * i + 1][6] = -data1[i][0] * data2[i][1]
            A[2 * i + 1][7] = -data1[i][1] * data2[i][1]
            B[2 * i] = data2[i][0]
            B[2 * i + 1] = data2[i][1]

        try:
            dot = np.dot
            AT = A.T
            X = dot(dot(linalg.inv(np.dot(AT, A)), AT), B)

        except Exception as e:
            print("Error when calcuating M: ", e)
            return False
        else:
            X.resize((3, 3), refcheck=False)
            X[2][2] = 1
        return X

    @classmethod
    def get_value(cls, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3)-> Tuple[int, float]:
        transformed = cls.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold

        if np.sum(good) > 0:
            centroid = np.sum(points1[good], axis=0) / np.sum(good)
            distance = np.sum(np.square(points1[good] - centroid))
            error = distance / np.sum(good)
        else:
            error = 0
        return np.sum(good), error

    def run(self):
        data = []
        for i in range(self.GENERATION):

            self.population = sorted(self.population, reverse=True)
            data.append([j.value for j in self.population[:min(4, len(self.population))]])
            self.population = self.population[:len(self.population) // 2]
        
            for j in range(min(12, 4 * len(self.population))):
                new = deepcopy(self.population[j // 3])
                rand_index = np.random.choice(range(8), self.MUTATION_COUNT, replace=0)
                gene = new.dna.take(rand_index)
                gene *= np.random.normal(1, .1, rand_index.size)
                new.dna.put(rand_index, gene)
                new.value = self.get_judgement(self.data1, self.data2, new.dna)
                self.population.append(new)

            children = []
            while len(children) + len(self.population) <= self.SAMPLE:
                mother = random.choice(self.population)
                father = random.choice(self.population)
                corss_index = random.choices(range(8), k=4)
                child = deepcopy(mother)
                child.dna.put([corss_index], father.dna.take([corss_index]))
                if random.random() < self.MUTATION_RATE:
                    rand_index = np.random.choice(range(8), self.MUTATION_COUNT, replace=0)
                    gene = child.dna.take(rand_index)
                    gene *= np.random.chisquare(2, rand_index.size)
                    child.dna.put(rand_index, gene)
                child.value = self.get_judgement(self.data1, self.data2, child.dna)
                children.append(child)

            self.population = np.concatenate((self.population, children))

        # 获取最优点
        self.log(np.array(data).T)
        self.population = sorted(self.population, reverse=True)
        if self.population[0].value > 3:
            self.good_points = self.population[0].value
            return self.population[0].dna
        else:
            raise RuntimeError("Cannot get transfrom matrix")



if(1):
    GeneticTransform = GeneticTransformWithPosition
    import warnings
    warnings.warn(
        "In this program, original ga algorithm is replaced with a exprimental one, to disable this, go to this file and comment the line")


def main():
    pass
    test()


def test():
    import time

    data_point1 = np.array([[1, 2], [3, 3], [5, 5], [6, 8]])
    data_point2 = np.array([[4, 2], [5, 3], [12, 5], [64, 8]])
    ransac = Ransac(data_point1, data_point2)
    M = ransac.get_perspective_transform(data_point1, data_point2)
    print(M)
    print("Supposed to be:\n",
          "[[-0.76850095,  1.15180266,  1.77229602]\n",
          "[ 0.20777989, -0.31593928,  2.07779886],\n",
          "[-0.10388994, -0.03462998,  1.        ]]")
    print(ransac.get_good_points(data_point1, data_point1, M))

    test_data = np.random.rand(40, 2) * 100
    dst_data = Ransac.perspective_transform(test_data, M)
    dst_data[:10, :] = np.random.rand(10, 2)
    start = time.time()
    ransac = Ransac(test_data, dst_data)
    result_M = ransac.run()
    print("Spent time: ", time.time() - start)
    print("Result M:")
    print(result_M)
    print("Max itereration times")
    print(ransac.max_iter_times)

    # Test for GA
    print("===== Test for ga =====")
    data1 = np.array([[520.12805, 243.64803], [4679.038, 627.0568], [508.80002, 277.2], [259.78067, 403.10794], [661.2, 199.20001], [695.13727, 232.90681],
                      [831.51373, 134.78401], [512.395, 243.65637], [329.65274, 756.0514], [502., 292.],
                      [523.2, 244.8], [522.72003, 244.8], [1870.3875, 1967.8467], [5208.884, 904.0897],
                      [661., 199.], [1872.2125, 1967.764], [1003.2909, 240.07318], [316., 736.], [
        612., 170.40001], [520.4737, 244.68483], [534.98895, 199.06564],
        [523., 245.], [582., 220.], [5072., 756.], [391.68002, 116.64001], [4572.2886, 1094.861], [1871.217, 1968.2616], [1003.2909, 238.87878]], dtype=np.float32)
    data2 = np.array([[1080.0001, 221.18402], [5297.137, 400.12195], [1069.2001, 253.20001], [845.0337, 394.15], [1218., 168.], [1250.5305, 200.65819],
                      [1374.797, 93.31201], [1074.9546, 222.15727], [910.1282, 730.9691],
                      [1093., 177.], [1082., 221.], [1081.4401, 220.32], [2401.2292, 1874.5347],
                      [708., 389.], [1218., 168.], [2403.7178, 1875.1985], [1540.7682, 186.32545], [
        901., 711.], [1153., 261.], [1078.2721, 221.87523], [1094.8611, 176.67076],
        [847., 44.], [1426., 175.], [1244.4, 326.40002], [823., 32.], [5173.633, 906.1633], [2401.2292, 1873.7054], [1540.7682, 185.13106]], dtype=np.float32)
    start = time.time()
    gr = GeneticRansac(data1, data2)
    print("Spent time: ", time.time() - start)
    print(gr.run())
    print("===== Test for new ga =====")
    start = time.time()
    gr = NewGeneticRansac(data1, data2)
    print("Spent time: ", time.time() - start)
    print(gr.run())
    print("===== Test for genetic tranform =====")
    print(GeneticTransform.get_lss_matrix(data_point1, data_point2))
    print(GeneticTransform(data1, data2).run())
    print(GeneticTransform.get_value(data_point1, data_point2, M))


if __name__ == "__main__":
    main()