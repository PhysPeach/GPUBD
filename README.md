# やりたいこと
1. 2次元の箱の中にたくさんの粒子を入れる。(箱は周期境界)
1. 毎ステップ粒子間の相互作用の計算を行って、そこで得た力を元に粒子を少しずつ動かす。
1. ある時間ごとの粒子座標を記録する。

# 仕様
## クラス
このコードには現在Particles, Grid, Boxの3つのクラスがある。
Particlesは「粒子配置、速度分布の決定」「運動エネルギーの解析」を担当する。※なお、相互作用に関連するものについてはGridへ移した。
Gridは「粒子間の関係性」に着目したクラスとした。従って「粒子が相互作用するペアの決定、相互作用力の決定」「位置エネルギーの解析」を担当する。
Boxは「ParticlesとBoxを結びつける」「時間発展させる」クラスとした。

## Particles
particlesはN個のD次元ガラス粒子群を表現するクラスである。ホスト側、デバイス側それぞれに変数と関数がある。

### ホスト側
- float diam[N]: 半径群
- double x[D*N]: 座標群
- float v[D*N]: 速度群

- void makeParticles(Particles* p): (デバイス側も含め)粒子群を作る
- void killParticles(Particles* p):（デバイス側も含め)粒子群をFreeする。
- void scatterParticles(Particles* p, double L): 粒子群を長さLの範囲に一様に散りばめる。

- void removevg2D(Particles* p): 粒子群の平均速度を0にする

- float K(Particles* particles): 運動エネルギーを解析する。

### デバイス側
- float diam_dev[]
- float x_dev[]
- flaot v_dev[]
- curandState *rndState_dev[D*N]: ランダム力
- float force_dev[D*N]: 外力(Gridによって計算される)

- __global__ void vEvo(float *v, double dt, float themalFuctor, float *force, curandState *state): この時点での粒子群の速度を更新する。
- __global__ xDvlp(double *x, double dt, double L, float *v): この時点での粒子群の座標を更新する。

## Grid
箱をgridによって分割した一つの単位をcellと呼ぶ
### ホスト側
- double rc：cellの1辺の大きさ
- uint M: gridの1辺に並ぶcellの数
- uint EpM: cell1つあたりに入る粒子の数
- uint updateFreq: gridの更新頻度

- void makeGrid(Grid* grid, double L): Gridを作る。
- void killGrid(Grid* grid): GridをFreeする。
- void makeCellPattern2D(Grid* grid): refCell_devを決定する。

- void setUpdateFreq(Grid*grid, double, dt, float *v): updateFreqを決定する。
- void checkUpdate(Grid* grid, double dt, double *x, float *v): gridを更新するか調べる。

- float U(Grid* grid, float *diam, double *x): 位置エネルギーを解析する。

### デバイス側
- uint* refCell_dev: blockが参照するcellの順序を記した数列
- uint cell_dev[(M^D)*EpM]: cellの実体

- float vmax_dev[2][]: 速度の最大値を求めるための配列
- float getNU_dev[2][]: 位置エネルギーを得るための配列

- __global__ void updateGrid(Grid grid, uint* cell, double *x): gridを更新する。
- __global__ void ~int2D系: 相互作用力を決定する。

## Box
- uint id: 箱のID
- double dt: 時間素子
- double t: 時間
- float T: 温度
- float thermalFuctor: ランダム力のスケール
- double L: 箱の1辺の長さ
- Particles: p: 粒子群の実体
- Grid g: gridの実体

- std::ofstream positionFile: 粒子群の記録(log プロット)
- std::ofstream animeFile: 粒子群の記録(線形)

- inline void setdt_T(Box* box, double setdt, float setT): dt, T, thermalFuctorの決定

- void prepareBox(Box* box): 箱を作る。
- void killBox(Box* box): 箱をFreeする。
- void prepareBox(Box* box): 箱に粒子を散りばめ、Gridを導入
- void initBox(Box* box): 記録の前処理

- inline void harmonicEvoBox(Box* box): 相互作用力にハーモニックポテンシャルを使って時間発展させる。
- inline void tEvoBox(Box* box): 系を時間発展させる。

- void equilibrateBox(Box* box, double teq): teqの時間をかけて系を平衡化させる。

- void recBox(std::ofstream *of, Box* box): *ofにその時点での粒子群の情報を記録

- void getData(Box* box): 系の時間発展を記録
- void benchmark(Box* box, uint loop): loop回時間発展を回す。

最終更新日 12/Nov./2019