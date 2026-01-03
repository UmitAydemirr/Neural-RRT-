import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import json
import os
import math
import glob
import scipy.interpolate as interp
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import multiprocessing


hedef_harita_sayisi = 10000
# hedef_harita_sayisi = 30
Max_iterasyon = 5000

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
class Harita:
    def __init__(self, width=500, height=500):
        self.width = random.randint(100, width)
        self.height = random.randint(100, height)
        self.engeller = []
        self.baslangic = None
        self.hedef = None
        self.engel_daire = []
        self.engel_dikdortgen = []
    def rastgele_engel_ekle(self):
        self.engeller = []
        self.engel_daire = []
        self.engel_dikdortgen = []
        alan = self.width * self.height
        yogunluk_katsayisi = 2000
        engel_sayisi = int(alan / yogunluk_katsayisi)
        engel_sayisi = max(5, engel_sayisi)
        for _ in range(engel_sayisi):
            if random.random() < 0.5:
                yaricap = random.randint(3, 15)
                x = random.randint(yaricap, self.width - yaricap)
                y = random.randint(yaricap, self.height - yaricap)
                self.engeller.append(('daire', x, y, yaricap))
                self.engel_daire.append((x, y, yaricap))
            else:
                genislik = random.randint(5, 20)
                yukseklik = random.randint(5, 20)
                x = random.randint(0, self.width - genislik)
                y = random.randint(0, self.height - yukseklik)
                self.engeller.append(('dikdortgen', x, y, genislik, yukseklik))
                self.engel_dikdortgen.append((x, y, genislik, yukseklik))
        self.engel_daire = np.array(self.engel_daire) if self.engel_daire else np.zeros((0, 3))
        self.engel_dikdortgen = np.array(self.engel_dikdortgen) if self.engel_dikdortgen else np.zeros((0, 4))

    def carpisma_kontrol(self,x,y, padding=5.0):
        if len(self.engel_daire) > 0:
            dx = self.engel_daire[:, 0] - x
            dy = self.engel_daire[:, 1] - y
            d2 = dx*dx + dy*dy
            guvenli_mesafe_karesi = (self.engel_daire[:, 2] + padding)**2
            if np.any(d2 <= guvenli_mesafe_karesi):
                return True
            
        if len(self.engel_dikdortgen) > 0:
            ox = self.engel_dikdortgen[:, 0]
            oy = self.engel_dikdortgen[:, 1]
            ow = self.engel_dikdortgen[:, 2]
            oh = self.engel_dikdortgen[:, 3]
            diktörgen_kontrol = (x >= ox - padding) & (x <= ox + ow + padding) & \
                                (y >= oy - padding) & (y <= oy + oh + padding)
            if np.any(diktörgen_kontrol):
                return True
        return False
    
    def kenar_carpisma_kontrol(self, x1, y1, x2, y2, padding=5.0):
        mesafe = math.hypot(x2 - x1, y2 - y1)
        adim_sayisi = 1
        if mesafe < adim_sayisi:
            return self.carpisma_kontrol(x2, y2, padding)
        gerekli_adim = int(mesafe / adim_sayisi)
        t = np.linspace(0, 1, gerekli_adim + 1)
        x_noktalari = x1 + t * (x2 - x1)
        y_noktalari = y1 + t * (y2 - y1)
        for i in range(len(x_noktalari)):
            if self.carpisma_kontrol(x_noktalari[i], y_noktalari[i], padding):
                return True
        return False
    def baslangic_bitis_olustur(self):
        deneme_limiti = 5000
        sayac = 0
        while sayac < deneme_limiti:
            baslangic = [random.uniform(5,self.width-5), random.uniform(5,self.height-5)]
            if not self.carpisma_kontrol(baslangic[0], baslangic[1], padding=5.0):
                self.baslangic = Node(baslangic[0], baslangic[1])
                break
            sayac += 1
        else:
            return False
        sayac = 0
        while sayac < deneme_limiti:
            bitis = [random.uniform(5,self.width-5), random.uniform(5,self.height-5)]
            mesafe = math.hypot(bitis[0]-self.baslangic.x, bitis[1]-self.baslangic.y)
            if not self.carpisma_kontrol(bitis[0], bitis[1], padding=5.0) and mesafe > 100:
                self.hedef = Node(bitis[0], bitis[1])
                break
            sayac += 1
        else:
            return False
    def data_kayıt(self, dosya_adi, rrt_yolu):
        data = {
            "harita_boyut": { "genislik": self.width, "yukseklik": self.height  },
            "baslangic": [self.baslangic.x, self.baslangic.y],
            "hedef": [self.hedef.x, self.hedef.y],
            "engeller": self.engeller,
            "rrt_path": rrt_yolu
        }
        with open(f"dataset/{dosya_adi}.json", "w") as f:
            json.dump(data, f)
        
    def harita_gorsel(self, dosya_adi, rrt_yolu):
        dpi = 100
        fig_x = self.width / dpi
        fig_y = self.height / dpi
        fig, ax = plt.subplots(figsize=(fig_x,fig_y) , dpi=dpi)
        ax.set_aspect('equal')

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        for engel in self.engeller:
            if engel[0] == 'daire':
                circle = patches.Circle((engel[1], engel[2]), engel[3], color='gray')
                ax.add_patch(circle)
            elif engel[0] == 'dikdortgen':
                rect = patches.Rectangle((engel[1], engel[2]), engel[3], engel[4], color='gray')
                ax.add_patch(rect)
        ax.plot(self.baslangic.x, self.baslangic.y, 'go',  markersize=8)
        ax.plot(self.hedef.x, self.hedef.y, 'rx', markersize=8)
        if rrt_yolu:
            yol = np.array(rrt_yolu)
            ax.plot(yol[:,0], yol[:,1], '-b', linewidth=1)
        plt.axis('off')
        plt.savefig(f"dataset/{dosya_adi}.png")
        plt.close(fig)

class BRRTStar:
    def __init__(self, harita, genisleme_mesafesi=4.0, max_iterasyon=5000, padding=5.0):
        self.env = harita
        self.genisleme_mesafesi = genisleme_mesafesi
        self.max_iterasyon = max_iterasyon
        self.padding = padding

    def planning(self):
        self.baslangic_tree = [self.env.baslangic]
        self.end_tree = [self.env.hedef]
        self.baslangic_coords = [[self.env.baslangic.x, self.env.baslangic.y]]
        self.end_coords = [[self.env.hedef.x, self.env.hedef.y]]
        self.baslangic_kdtree = cKDTree(self.baslangic_coords)
        self.end_kdtree = cKDTree(self.end_coords)
        min_total_cost = float("inf")
        best_path = None
        for i in range(self.max_iterasyon):
            rnd_node = self.get_random_node()
            if i % 2 == 0:
                tree_a, coords_a, kdtree_a = self.baslangic_tree, self.baslangic_coords, self.baslangic_kdtree
                tree_b, coords_b, kdtree_b = self.end_tree, self.end_coords, self.end_kdtree
                is_forward = True
            else:
                tree_a, coords_a, kdtree_a = self.end_tree, self.end_coords, self.end_kdtree
                tree_b, coords_b, kdtree_b = self.baslangic_tree, self.baslangic_coords, self.baslangic_kdtree
                is_forward = False

            dist, idx = kdtree_a.query([rnd_node.x, rnd_node.y])
            nearest_node_a = tree_a[idx]
            new_node_a = self.steer(nearest_node_a, rnd_node, self.genisleme_mesafesi)

            if not self.env.kenar_carpisma_kontrol(nearest_node_a.x, nearest_node_a.y, new_node_a.x, new_node_a.y, padding=self.padding):
                near_inds_a = kdtree_a.query_ball_point([new_node_a.x, new_node_a.y], r=20.0)
                new_node_a = self.choose_parent(tree_a, new_node_a, near_inds_a)
                if new_node_a:
                    tree_a.append(new_node_a)
                    coords_a.append([new_node_a.x, new_node_a.y])
                    if i % 2 == 0: self.baslangic_kdtree = cKDTree(coords_a)
                    else: self.end_kdtree = cKDTree(coords_a)
                    self.rewire(tree_a, new_node_a, near_inds_a)
                    dist_b, idx_b = kdtree_b.query([new_node_a.x, new_node_a.y])
                    nearest_node_b = tree_b[idx_b]
                    if dist_b <= self.genisleme_mesafesi:
                        if not self.env.kenar_carpisma_kontrol(new_node_a.x, new_node_a.y, nearest_node_b.x, nearest_node_b.y, padding=self.padding):
                            current_total_cost = new_node_a.cost + nearest_node_b.cost + dist_b
                            if current_total_cost < min_total_cost:
                                min_total_cost = current_total_cost
                                if is_forward: temp_path = self.generate_path(new_node_a, nearest_node_b)
                                else: temp_path = self.generate_path(nearest_node_b, new_node_a)
                                best_path = self.apply_bspline_smoothing(temp_path)
                                return best_path
        return best_path

    def apply_bspline_smoothing(self, path):

        if not path or len(path) < 3: 
            return path
            
        try:
            path = np.array(path)
            
            unique_path = [path[0]]
            for i in range(1, len(path)):
                dist = np.linalg.norm(path[i] - path[i-1])
                if dist > 0.2:
                    unique_path.append(path[i])
            
   
            if np.linalg.norm(np.array(unique_path[-1]) - path[-1]) > 0.001:
                unique_path.append(path[-1])
                
            path = np.array(unique_path)
            
            if len(path) < 3: 
                return path.tolist()

            x, y = path[:, 0], path[:, 1]
            
            degree = min(2, len(path) - 1) 
            
            tck, u = interp.splprep([x, y], s=0.0, k=degree)
            
            total_length = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
            num_points = int(total_length / 0.5) 
            num_points = max(100, num_points)
            
            u_new = np.linspace(0, 1, num=num_points)
            x_new, y_new = interp.splev(u_new, tck)
            smooth_path = list(zip(x_new, y_new))
            
            guvenlik_payi = self.padding * 1.2 
            
            for p in smooth_path:

                if self.env.carpisma_kontrol(p[0], p[1], padding=guvenlik_payi):
                    return path.tolist()
            

            return [list(p) for p in smooth_path]

        except Exception as e:

            return path.tolist()

    def generate_path(self, node_baslangic_side, node_end_side):
        path_baslangic = []
        curr = node_baslangic_side
        while curr.parent:
            path_baslangic.append([curr.x, curr.y])
            curr = curr.parent
        path_baslangic.append([curr.x, curr.y])
        path_baslangic = path_baslangic[::-1]
        path_end = []
        curr = node_end_side
        while curr.parent:
            path_end.append([curr.x, curr.y])
            curr = curr.parent
        path_end.append([curr.x, curr.y])
        return path_baslangic + path_end

    def get_random_node(self):
        return Node(random.uniform(0, self.env.width), random.uniform(0, self.env.height))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.cost = from_node.cost + extend_length
        new_node.parent = from_node
        return new_node

    def choose_parent(self, node_list, new_node, near_inds):
        if not near_inds: return None
        costs = []
        valid_inds = []
        for i in near_inds:
            near_node = node_list[i]
            if not self.env.kenar_carpisma_kontrol(near_node.x, near_node.y, new_node.x, new_node.y, padding=self.padding):
                costs.append(near_node.cost + self.calc_dist(near_node, new_node))
                valid_inds.append(i)
        if not costs: return None
        min_cost = min(costs)
        min_ind = valid_inds[costs.index(min_cost)]
        new_node.parent = node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, node_list, new_node, near_inds):
        for i in near_inds:
            near_node = node_list[i]
            if not self.env.kenar_carpisma_kontrol(new_node.x, new_node.y, near_node.x, near_node.y, padding=self.padding):
                new_cost = new_node.cost + self.calc_dist(new_node, near_node)
                if near_node.cost > new_cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    @staticmethod
    def calc_dist(n1, n2): return math.hypot(n1.x - n2.x, n1.y - n2.y)

def process_single_map(map_id):
    try:
        env = Harita()
        env.rastgele_engel_ekle()
        env.baslangic_bitis_olustur()
        solver = BRRTStar(env, max_iterasyon=Max_iterasyon, padding=5.0)
        path = solver.planning()
        if path is not None:
            filename = f"map_{map_id}"
            env.data_kayıt(filename, path)
            # if hedef_harita_sayisi == 10000:
            #     if map_id % 500 == 0: 
            #         env.harita_gorsel(filename, path)
            # elif hedef_harita_sayisi <= 100:
            #     env.harita_gorsel(filename, path)
            # else:
            #     if map_id % 50 == 0:
            #         env.harita_gorsel(filename, path)

            return True
        else:
            return False
    except Exception as e:
        print(f"Error processing map {map_id}: {e}")
        return False

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    cekirdek_sayisi = multiprocessing.cpu_count()
    kullanilacak_cekirdek_sayisi = max(1, cekirdek_sayisi//2) 
    
    print(f"Cekirdek Sayisi: {cekirdek_sayisi}")
    print(f"Kullanilacak Cekirdek Sayisi: {kullanilacak_cekirdek_sayisi}")

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    dosyalar = glob.glob("dataset/map_*.json")
    
    mevcut_dosya_sayisi = len(dosyalar)
    
    en_son_id = 0
    if dosyalar:
        icerik = [int(os.path.basename(f).replace("map_", "").replace(".json", "")) for f in dosyalar if "map_" in f]
        if icerik: 
            en_son_id = max(icerik)
    
    baslangic_id = en_son_id + 1

    print(f"Mevcut Dosya Sayisi (Adet): {mevcut_dosya_sayisi}")
    print(f"En Son Dosya ID'si: map_{en_son_id}")
    print(f"Yeni Dosyalar map_{baslangic_id} isminden baslayacak.")
    print(f"Hedef Harita Sayisi: {hedef_harita_sayisi}")

    gerekli_sayi = hedef_harita_sayisi - mevcut_dosya_sayisi

    if gerekli_sayi > 0:
        print(f"Gereken Harita Sayisi: {gerekli_sayi} adet üretilecek...")
        
        aralik = range(baslangic_id, baslangic_id + gerekli_sayi)
        
        sonuclar = Parallel(n_jobs=kullanilacak_cekirdek_sayisi, verbose=5)(
            delayed(process_single_map)(i) for i in aralik
        )
        
        basari_sayisi = sonuclar.count(True)
        
        toplam_var_olan = mevcut_dosya_sayisi + basari_sayisi
        print(f"Bu turda basariyla olusturulan: {basari_sayisi}")
        print(f"Toplam Veri Seti Boyutu: {toplam_var_olan}/{hedef_harita_sayisi}")
        
        if toplam_var_olan < hedef_harita_sayisi:
            print("Not: Hatalı haritalar nedeniyle hedef sayiya ulasilamadi. Programi tekrar çalıştır.")
    else:
        print("Hedef harita sayisina zaten ulasilmis veya gecilmis.")