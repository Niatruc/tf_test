       �K"	  ��6B�Abrain.Event:2G���a      �.��	0^��6B�A"��
q
inputs/x_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
inputs/y_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
t
#zlayer1/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
g
"zlayer1/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$zlayer1/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2zlayer1/weights/random_normal/RandomStandardNormalRandomStandardNormal#zlayer1/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
!zlayer1/weights/random_normal/mulMul2zlayer1/weights/random_normal/RandomStandardNormal$zlayer1/weights/random_normal/stddev*
_output_shapes

:
*
T0
�
zlayer1/weights/random_normalAdd!zlayer1/weights/random_normal/mul"zlayer1/weights/random_normal/mean*
T0*
_output_shapes

:

�
zlayer1/weights/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
zlayer1/weights/Variable/AssignAssignzlayer1/weights/Variablezlayer1/weights/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@zlayer1/weights/Variable
�
zlayer1/weights/Variable/readIdentityzlayer1/weights/Variable*
T0*+
_class!
loc:@zlayer1/weights/Variable*
_output_shapes

:

u
zlayer1/weights/Weights1/tagConst*)
value B Bzlayer1/weights/Weights1*
dtype0*
_output_shapes
: 
�
zlayer1/weights/Weights1HistogramSummaryzlayer1/weights/Weights1/tagzlayer1/weights/Variable/read*
T0*
_output_shapes
: 
i
zlayer1/biases/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

Y
zlayer1/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
n
zlayer1/biases/addAddzlayer1/biases/zeroszlayer1/biases/add/y*
_output_shapes

:
*
T0
�
zlayer1/biases/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
zlayer1/biases/Variable/AssignAssignzlayer1/biases/Variablezlayer1/biases/add*
use_locking(*
T0**
_class 
loc:@zlayer1/biases/Variable*
validate_shape(*
_output_shapes

:

�
zlayer1/biases/Variable/readIdentityzlayer1/biases/Variable*
_output_shapes

:
*
T0**
_class 
loc:@zlayer1/biases/Variable
q
zlayer1/biases/biases1/tagConst*'
valueB Bzlayer1/biases/biases1*
dtype0*
_output_shapes
: 
�
zlayer1/biases/biases1HistogramSummaryzlayer1/biases/biases1/tagzlayer1/biases/Variable/read*
T0*
_output_shapes
: 
�
zlayer1/Wx_plus_b/MatMulMatMulinputs/x_inputzlayer1/weights/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
zlayer1/Wx_plus_b/addAddzlayer1/Wx_plus_b/MatMulzlayer1/biases/Variable/read*
T0*'
_output_shapes
:���������

]
zlayer1/ReluReluzlayer1/Wx_plus_b/add*
T0*'
_output_shapes
:���������

e
zlayer1/outputs1/tagConst*
dtype0*
_output_shapes
: *!
valueB Bzlayer1/outputs1
i
zlayer1/outputs1HistogramSummaryzlayer1/outputs1/tagzlayer1/Relu*
T0*
_output_shapes
: 
t
#zlayer2/weights/random_normal/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
"zlayer2/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$zlayer2/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2zlayer2/weights/random_normal/RandomStandardNormalRandomStandardNormal#zlayer2/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
!zlayer2/weights/random_normal/mulMul2zlayer2/weights/random_normal/RandomStandardNormal$zlayer2/weights/random_normal/stddev*
T0*
_output_shapes

:

�
zlayer2/weights/random_normalAdd!zlayer2/weights/random_normal/mul"zlayer2/weights/random_normal/mean*
_output_shapes

:
*
T0
�
zlayer2/weights/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
zlayer2/weights/Variable/AssignAssignzlayer2/weights/Variablezlayer2/weights/random_normal*
use_locking(*
T0*+
_class!
loc:@zlayer2/weights/Variable*
validate_shape(*
_output_shapes

:

�
zlayer2/weights/Variable/readIdentityzlayer2/weights/Variable*
_output_shapes

:
*
T0*+
_class!
loc:@zlayer2/weights/Variable
u
zlayer2/weights/Weights2/tagConst*)
value B Bzlayer2/weights/Weights2*
dtype0*
_output_shapes
: 
�
zlayer2/weights/Weights2HistogramSummaryzlayer2/weights/Weights2/tagzlayer2/weights/Variable/read*
T0*
_output_shapes
: 
i
zlayer2/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes

:
Y
zlayer2/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
n
zlayer2/biases/addAddzlayer2/biases/zeroszlayer2/biases/add/y*
T0*
_output_shapes

:
�
zlayer2/biases/Variable
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
zlayer2/biases/Variable/AssignAssignzlayer2/biases/Variablezlayer2/biases/add*
use_locking(*
T0**
_class 
loc:@zlayer2/biases/Variable*
validate_shape(*
_output_shapes

:
�
zlayer2/biases/Variable/readIdentityzlayer2/biases/Variable*
T0**
_class 
loc:@zlayer2/biases/Variable*
_output_shapes

:
q
zlayer2/biases/biases2/tagConst*'
valueB Bzlayer2/biases/biases2*
dtype0*
_output_shapes
: 
�
zlayer2/biases/biases2HistogramSummaryzlayer2/biases/biases2/tagzlayer2/biases/Variable/read*
T0*
_output_shapes
: 
�
zlayer2/Wx_plus_b/MatMulMatMulzlayer1/Reluzlayer2/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
zlayer2/Wx_plus_b/addAddzlayer2/Wx_plus_b/MatMulzlayer2/biases/Variable/read*
T0*'
_output_shapes
:���������
e
zlayer2/outputs2/tagConst*
dtype0*
_output_shapes
: *!
valueB Bzlayer2/outputs2
r
zlayer2/outputs2HistogramSummaryzlayer2/outputs2/tagzlayer2/Wx_plus_b/add*
T0*
_output_shapes
: 
h
loss/subSubinputs/y_inputzlayer2/Wx_plus_b/add*
T0*'
_output_shapes
:���������
Q
loss/SquareSquareloss/sub*'
_output_shapes
:���������*
T0
d
loss/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
loss/SumSumloss/Squareloss/Sum/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/Sum
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
loss/loss/tagsConst*
valueB B	loss/loss*
dtype0*
_output_shapes
: 
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/Sum*
_output_shapes
:*
T0*
out_type0
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Sum*
_output_shapes
:*
T0*
out_type0
i
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
n
#train/gradients/loss/Sum_grad/ShapeShapeloss/Square*
T0*
out_type0*
_output_shapes
:
�
"train/gradients/loss/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
!train/gradients/loss/Sum_grad/addAddloss/Sum/reduction_indices"train/gradients/loss/Sum_grad/Size*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
!train/gradients/loss/Sum_grad/modFloorMod!train/gradients/loss/Sum_grad/add"train/gradients/loss/Sum_grad/Size*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/Shape_1Const*
valueB:*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
:
�
)train/gradients/loss/Sum_grad/range/startConst*
value	B : *6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
)train/gradients/loss/Sum_grad/range/deltaConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
#train/gradients/loss/Sum_grad/rangeRange)train/gradients/loss/Sum_grad/range/start"train/gradients/loss/Sum_grad/Size)train/gradients/loss/Sum_grad/range/delta*

Tidx0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
(train/gradients/loss/Sum_grad/Fill/valueConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
"train/gradients/loss/Sum_grad/FillFill%train/gradients/loss/Sum_grad/Shape_1(train/gradients/loss/Sum_grad/Fill/value*
T0*

index_type0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
+train/gradients/loss/Sum_grad/DynamicStitchDynamicStitch#train/gradients/loss/Sum_grad/range!train/gradients/loss/Sum_grad/mod#train/gradients/loss/Sum_grad/Shape"train/gradients/loss/Sum_grad/Fill*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
N*
_output_shapes
:
�
'train/gradients/loss/Sum_grad/Maximum/yConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
%train/gradients/loss/Sum_grad/MaximumMaximum+train/gradients/loss/Sum_grad/DynamicStitch'train/gradients/loss/Sum_grad/Maximum/y*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
&train/gradients/loss/Sum_grad/floordivFloorDiv#train/gradients/loss/Sum_grad/Shape%train/gradients/loss/Sum_grad/Maximum*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv+train/gradients/loss/Sum_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
"train/gradients/loss/Sum_grad/TileTile%train/gradients/loss/Sum_grad/Reshape&train/gradients/loss/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
&train/gradients/loss/Square_grad/ConstConst#^train/gradients/loss/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Square_grad/MulMulloss/sub&train/gradients/loss/Square_grad/Const*
T0*'
_output_shapes
:���������
�
&train/gradients/loss/Square_grad/Mul_1Mul"train/gradients/loss/Sum_grad/Tile$train/gradients/loss/Square_grad/Mul*
T0*'
_output_shapes
:���������
q
#train/gradients/loss/sub_grad/ShapeShapeinputs/y_input*
T0*
out_type0*
_output_shapes
:
z
%train/gradients/loss/sub_grad/Shape_1Shapezlayer2/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/sub_grad/Shape%train/gradients/loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/loss/sub_grad/SumSum&train/gradients/loss/Square_grad/Mul_13train/gradients/loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
%train/gradients/loss/sub_grad/ReshapeReshape!train/gradients/loss/sub_grad/Sum#train/gradients/loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
#train/gradients/loss/sub_grad/Sum_1Sum&train/gradients/loss/Square_grad/Mul_15train/gradients/loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
p
!train/gradients/loss/sub_grad/NegNeg#train/gradients/loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
'train/gradients/loss/sub_grad/Reshape_1Reshape!train/gradients/loss/sub_grad/Neg%train/gradients/loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
.train/gradients/loss/sub_grad/tuple/group_depsNoOp&^train/gradients/loss/sub_grad/Reshape(^train/gradients/loss/sub_grad/Reshape_1
�
6train/gradients/loss/sub_grad/tuple/control_dependencyIdentity%train/gradients/loss/sub_grad/Reshape/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape*'
_output_shapes
:���������
�
8train/gradients/loss/sub_grad/tuple/control_dependency_1Identity'train/gradients/loss/sub_grad/Reshape_1/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/sub_grad/Reshape_1*'
_output_shapes
:���������
�
0train/gradients/zlayer2/Wx_plus_b/add_grad/ShapeShapezlayer2/Wx_plus_b/MatMul*
_output_shapes
:*
T0*
out_type0
�
2train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
@train/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs0train/gradients/zlayer2/Wx_plus_b/add_grad/Shape2train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
.train/gradients/zlayer2/Wx_plus_b/add_grad/SumSum8train/gradients/loss/sub_grad/tuple/control_dependency_1@train/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2train/gradients/zlayer2/Wx_plus_b/add_grad/ReshapeReshape.train/gradients/zlayer2/Wx_plus_b/add_grad/Sum0train/gradients/zlayer2/Wx_plus_b/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
0train/gradients/zlayer2/Wx_plus_b/add_grad/Sum_1Sum8train/gradients/loss/sub_grad/tuple/control_dependency_1Btrain/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1Reshape0train/gradients/zlayer2/Wx_plus_b/add_grad/Sum_12train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
;train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_depsNoOp3^train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape5^train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1
�
Ctrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependencyIdentity2train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape<^train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape*'
_output_shapes
:���������
�
Etrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency_1Identity4train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1<^train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_deps*
_output_shapes

:*
T0*G
_class=
;9loc:@train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1
�
4train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMulMatMulCtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependencyzlayer2/weights/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(
�
6train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1MatMulzlayer1/ReluCtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
>train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp5^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul7^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1
�
Ftrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity4train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul?^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*G
_class=
;9loc:@train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul
�
Htrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity6train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1?^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1*
_output_shapes

:

�
*train/gradients/zlayer1/Relu_grad/ReluGradReluGradFtrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependencyzlayer1/Relu*
T0*'
_output_shapes
:���������

�
0train/gradients/zlayer1/Wx_plus_b/add_grad/ShapeShapezlayer1/Wx_plus_b/MatMul*
T0*
out_type0*
_output_shapes
:
�
2train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
@train/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs0train/gradients/zlayer1/Wx_plus_b/add_grad/Shape2train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
.train/gradients/zlayer1/Wx_plus_b/add_grad/SumSum*train/gradients/zlayer1/Relu_grad/ReluGrad@train/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2train/gradients/zlayer1/Wx_plus_b/add_grad/ReshapeReshape.train/gradients/zlayer1/Wx_plus_b/add_grad/Sum0train/gradients/zlayer1/Wx_plus_b/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
0train/gradients/zlayer1/Wx_plus_b/add_grad/Sum_1Sum*train/gradients/zlayer1/Relu_grad/ReluGradBtrain/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1Reshape0train/gradients/zlayer1/Wx_plus_b/add_grad/Sum_12train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
;train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_depsNoOp3^train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape5^train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1
�
Ctrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependencyIdentity2train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape<^train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape*'
_output_shapes
:���������

�
Etrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency_1Identity4train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1<^train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1*
_output_shapes

:

�
4train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMulMatMulCtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependencyzlayer1/weights/Variable/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
6train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1MatMulinputs/x_inputCtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
>train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp5^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul7^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1
�
Ftrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity4train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul?^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Htrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity6train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1?^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1*
_output_shapes

:

h
#train/GradientDescent/learning_rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
Jtrain/GradientDescent/update_zlayer1/weights/Variable/ApplyGradientDescentApplyGradientDescentzlayer1/weights/Variable#train/GradientDescent/learning_rateHtrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
T0*+
_class!
loc:@zlayer1/weights/Variable*
_output_shapes

:
*
use_locking( 
�
Itrain/GradientDescent/update_zlayer1/biases/Variable/ApplyGradientDescentApplyGradientDescentzlayer1/biases/Variable#train/GradientDescent/learning_rateEtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency_1*
T0**
_class 
loc:@zlayer1/biases/Variable*
_output_shapes

:
*
use_locking( 
�
Jtrain/GradientDescent/update_zlayer2/weights/Variable/ApplyGradientDescentApplyGradientDescentzlayer2/weights/Variable#train/GradientDescent/learning_rateHtrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@zlayer2/weights/Variable*
_output_shapes

:

�
Itrain/GradientDescent/update_zlayer2/biases/Variable/ApplyGradientDescentApplyGradientDescentzlayer2/biases/Variable#train/GradientDescent/learning_rateEtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency_1*
T0**
_class 
loc:@zlayer2/biases/Variable*
_output_shapes

:*
use_locking( 
�
train/GradientDescentNoOpJ^train/GradientDescent/update_zlayer1/biases/Variable/ApplyGradientDescentK^train/GradientDescent/update_zlayer1/weights/Variable/ApplyGradientDescentJ^train/GradientDescent/update_zlayer2/biases/Variable/ApplyGradientDescentK^train/GradientDescent/update_zlayer2/weights/Variable/ApplyGradientDescent
�
initNoOp^zlayer1/biases/Variable/Assign ^zlayer1/weights/Variable/Assign^zlayer2/biases/Variable/Assign ^zlayer2/weights/Variable/Assign"�gr      H��v	���6B�AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.13.12b'v1.13.0-rc2-5-g6612da8'��
q
inputs/x_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
inputs/y_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
t
#zlayer1/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
g
"zlayer1/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$zlayer1/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2zlayer1/weights/random_normal/RandomStandardNormalRandomStandardNormal#zlayer1/weights/random_normal/shape*
dtype0*
_output_shapes

:
*
seed2 *

seed *
T0
�
!zlayer1/weights/random_normal/mulMul2zlayer1/weights/random_normal/RandomStandardNormal$zlayer1/weights/random_normal/stddev*
_output_shapes

:
*
T0
�
zlayer1/weights/random_normalAdd!zlayer1/weights/random_normal/mul"zlayer1/weights/random_normal/mean*
T0*
_output_shapes

:

�
zlayer1/weights/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
zlayer1/weights/Variable/AssignAssignzlayer1/weights/Variablezlayer1/weights/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@zlayer1/weights/Variable
�
zlayer1/weights/Variable/readIdentityzlayer1/weights/Variable*
_output_shapes

:
*
T0*+
_class!
loc:@zlayer1/weights/Variable
u
zlayer1/weights/Weights1/tagConst*
dtype0*
_output_shapes
: *)
value B Bzlayer1/weights/Weights1
�
zlayer1/weights/Weights1HistogramSummaryzlayer1/weights/Weights1/tagzlayer1/weights/Variable/read*
T0*
_output_shapes
: 
i
zlayer1/biases/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

Y
zlayer1/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
n
zlayer1/biases/addAddzlayer1/biases/zeroszlayer1/biases/add/y*
_output_shapes

:
*
T0
�
zlayer1/biases/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
zlayer1/biases/Variable/AssignAssignzlayer1/biases/Variablezlayer1/biases/add*
use_locking(*
T0**
_class 
loc:@zlayer1/biases/Variable*
validate_shape(*
_output_shapes

:

�
zlayer1/biases/Variable/readIdentityzlayer1/biases/Variable*
T0**
_class 
loc:@zlayer1/biases/Variable*
_output_shapes

:

q
zlayer1/biases/biases1/tagConst*
dtype0*
_output_shapes
: *'
valueB Bzlayer1/biases/biases1
�
zlayer1/biases/biases1HistogramSummaryzlayer1/biases/biases1/tagzlayer1/biases/Variable/read*
_output_shapes
: *
T0
�
zlayer1/Wx_plus_b/MatMulMatMulinputs/x_inputzlayer1/weights/Variable/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
�
zlayer1/Wx_plus_b/addAddzlayer1/Wx_plus_b/MatMulzlayer1/biases/Variable/read*'
_output_shapes
:���������
*
T0
]
zlayer1/ReluReluzlayer1/Wx_plus_b/add*
T0*'
_output_shapes
:���������

e
zlayer1/outputs1/tagConst*
dtype0*
_output_shapes
: *!
valueB Bzlayer1/outputs1
i
zlayer1/outputs1HistogramSummaryzlayer1/outputs1/tagzlayer1/Relu*
T0*
_output_shapes
: 
t
#zlayer2/weights/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
g
"zlayer2/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$zlayer2/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2zlayer2/weights/random_normal/RandomStandardNormalRandomStandardNormal#zlayer2/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
!zlayer2/weights/random_normal/mulMul2zlayer2/weights/random_normal/RandomStandardNormal$zlayer2/weights/random_normal/stddev*
T0*
_output_shapes

:

�
zlayer2/weights/random_normalAdd!zlayer2/weights/random_normal/mul"zlayer2/weights/random_normal/mean*
T0*
_output_shapes

:

�
zlayer2/weights/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
zlayer2/weights/Variable/AssignAssignzlayer2/weights/Variablezlayer2/weights/random_normal*
use_locking(*
T0*+
_class!
loc:@zlayer2/weights/Variable*
validate_shape(*
_output_shapes

:

�
zlayer2/weights/Variable/readIdentityzlayer2/weights/Variable*
T0*+
_class!
loc:@zlayer2/weights/Variable*
_output_shapes

:

u
zlayer2/weights/Weights2/tagConst*)
value B Bzlayer2/weights/Weights2*
dtype0*
_output_shapes
: 
�
zlayer2/weights/Weights2HistogramSummaryzlayer2/weights/Weights2/tagzlayer2/weights/Variable/read*
_output_shapes
: *
T0
i
zlayer2/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes

:
Y
zlayer2/biases/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���=
n
zlayer2/biases/addAddzlayer2/biases/zeroszlayer2/biases/add/y*
_output_shapes

:*
T0
�
zlayer2/biases/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
zlayer2/biases/Variable/AssignAssignzlayer2/biases/Variablezlayer2/biases/add*
use_locking(*
T0**
_class 
loc:@zlayer2/biases/Variable*
validate_shape(*
_output_shapes

:
�
zlayer2/biases/Variable/readIdentityzlayer2/biases/Variable*
_output_shapes

:*
T0**
_class 
loc:@zlayer2/biases/Variable
q
zlayer2/biases/biases2/tagConst*
dtype0*
_output_shapes
: *'
valueB Bzlayer2/biases/biases2
�
zlayer2/biases/biases2HistogramSummaryzlayer2/biases/biases2/tagzlayer2/biases/Variable/read*
T0*
_output_shapes
: 
�
zlayer2/Wx_plus_b/MatMulMatMulzlayer1/Reluzlayer2/weights/Variable/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
zlayer2/Wx_plus_b/addAddzlayer2/Wx_plus_b/MatMulzlayer2/biases/Variable/read*
T0*'
_output_shapes
:���������
e
zlayer2/outputs2/tagConst*!
valueB Bzlayer2/outputs2*
dtype0*
_output_shapes
: 
r
zlayer2/outputs2HistogramSummaryzlayer2/outputs2/tagzlayer2/Wx_plus_b/add*
_output_shapes
: *
T0
h
loss/subSubinputs/y_inputzlayer2/Wx_plus_b/add*'
_output_shapes
:���������*
T0
Q
loss/SquareSquareloss/sub*
T0*'
_output_shapes
:���������
d
loss/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
loss/SumSumloss/Squareloss/Sum/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
T

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
e
	loss/MeanMeanloss/Sum
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
loss/loss/tagsConst*
dtype0*
_output_shapes
: *
valueB B	loss/loss
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/Sum*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Sum*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*#
_output_shapes
:���������*
T0
n
#train/gradients/loss/Sum_grad/ShapeShapeloss/Square*
T0*
out_type0*
_output_shapes
:
�
"train/gradients/loss/Sum_grad/SizeConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
!train/gradients/loss/Sum_grad/addAddloss/Sum/reduction_indices"train/gradients/loss/Sum_grad/Size*
_output_shapes
:*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
!train/gradients/loss/Sum_grad/modFloorMod!train/gradients/loss/Sum_grad/add"train/gradients/loss/Sum_grad/Size*
_output_shapes
:*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
%train/gradients/loss/Sum_grad/Shape_1Const*
valueB:*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
:
�
)train/gradients/loss/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
)train/gradients/loss/Sum_grad/range/deltaConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
#train/gradients/loss/Sum_grad/rangeRange)train/gradients/loss/Sum_grad/range/start"train/gradients/loss/Sum_grad/Size)train/gradients/loss/Sum_grad/range/delta*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:*

Tidx0
�
(train/gradients/loss/Sum_grad/Fill/valueConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
"train/gradients/loss/Sum_grad/FillFill%train/gradients/loss/Sum_grad/Shape_1(train/gradients/loss/Sum_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
+train/gradients/loss/Sum_grad/DynamicStitchDynamicStitch#train/gradients/loss/Sum_grad/range!train/gradients/loss/Sum_grad/mod#train/gradients/loss/Sum_grad/Shape"train/gradients/loss/Sum_grad/Fill*
N*
_output_shapes
:*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
'train/gradients/loss/Sum_grad/Maximum/yConst*
value	B :*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
%train/gradients/loss/Sum_grad/MaximumMaximum+train/gradients/loss/Sum_grad/DynamicStitch'train/gradients/loss/Sum_grad/Maximum/y*
_output_shapes
:*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
&train/gradients/loss/Sum_grad/floordivFloorDiv#train/gradients/loss/Sum_grad/Shape%train/gradients/loss/Sum_grad/Maximum*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv+train/gradients/loss/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
"train/gradients/loss/Sum_grad/TileTile%train/gradients/loss/Sum_grad/Reshape&train/gradients/loss/Sum_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
�
&train/gradients/loss/Square_grad/ConstConst#^train/gradients/loss/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Square_grad/MulMulloss/sub&train/gradients/loss/Square_grad/Const*'
_output_shapes
:���������*
T0
�
&train/gradients/loss/Square_grad/Mul_1Mul"train/gradients/loss/Sum_grad/Tile$train/gradients/loss/Square_grad/Mul*
T0*'
_output_shapes
:���������
q
#train/gradients/loss/sub_grad/ShapeShapeinputs/y_input*
T0*
out_type0*
_output_shapes
:
z
%train/gradients/loss/sub_grad/Shape_1Shapezlayer2/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/sub_grad/Shape%train/gradients/loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
!train/gradients/loss/sub_grad/SumSum&train/gradients/loss/Square_grad/Mul_13train/gradients/loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
%train/gradients/loss/sub_grad/ReshapeReshape!train/gradients/loss/sub_grad/Sum#train/gradients/loss/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
#train/gradients/loss/sub_grad/Sum_1Sum&train/gradients/loss/Square_grad/Mul_15train/gradients/loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
p
!train/gradients/loss/sub_grad/NegNeg#train/gradients/loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
'train/gradients/loss/sub_grad/Reshape_1Reshape!train/gradients/loss/sub_grad/Neg%train/gradients/loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
.train/gradients/loss/sub_grad/tuple/group_depsNoOp&^train/gradients/loss/sub_grad/Reshape(^train/gradients/loss/sub_grad/Reshape_1
�
6train/gradients/loss/sub_grad/tuple/control_dependencyIdentity%train/gradients/loss/sub_grad/Reshape/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape*'
_output_shapes
:���������
�
8train/gradients/loss/sub_grad/tuple/control_dependency_1Identity'train/gradients/loss/sub_grad/Reshape_1/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/sub_grad/Reshape_1*'
_output_shapes
:���������
�
0train/gradients/zlayer2/Wx_plus_b/add_grad/ShapeShapezlayer2/Wx_plus_b/MatMul*
T0*
out_type0*
_output_shapes
:
�
2train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
@train/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs0train/gradients/zlayer2/Wx_plus_b/add_grad/Shape2train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
.train/gradients/zlayer2/Wx_plus_b/add_grad/SumSum8train/gradients/loss/sub_grad/tuple/control_dependency_1@train/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2train/gradients/zlayer2/Wx_plus_b/add_grad/ReshapeReshape.train/gradients/zlayer2/Wx_plus_b/add_grad/Sum0train/gradients/zlayer2/Wx_plus_b/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
0train/gradients/zlayer2/Wx_plus_b/add_grad/Sum_1Sum8train/gradients/loss/sub_grad/tuple/control_dependency_1Btrain/gradients/zlayer2/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1Reshape0train/gradients/zlayer2/Wx_plus_b/add_grad/Sum_12train/gradients/zlayer2/Wx_plus_b/add_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
;train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_depsNoOp3^train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape5^train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1
�
Ctrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependencyIdentity2train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape<^train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape*'
_output_shapes
:���������
�
Etrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency_1Identity4train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1<^train/gradients/zlayer2/Wx_plus_b/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/zlayer2/Wx_plus_b/add_grad/Reshape_1*
_output_shapes

:
�
4train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMulMatMulCtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependencyzlayer2/weights/Variable/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
6train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1MatMulzlayer1/ReluCtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
�
>train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp5^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul7^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1
�
Ftrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity4train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul?^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul*'
_output_shapes
:���������

�
Htrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity6train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1?^train/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/zlayer2/Wx_plus_b/MatMul_grad/MatMul_1*
_output_shapes

:

�
*train/gradients/zlayer1/Relu_grad/ReluGradReluGradFtrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependencyzlayer1/Relu*
T0*'
_output_shapes
:���������

�
0train/gradients/zlayer1/Wx_plus_b/add_grad/ShapeShapezlayer1/Wx_plus_b/MatMul*
T0*
out_type0*
_output_shapes
:
�
2train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
@train/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs0train/gradients/zlayer1/Wx_plus_b/add_grad/Shape2train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
.train/gradients/zlayer1/Wx_plus_b/add_grad/SumSum*train/gradients/zlayer1/Relu_grad/ReluGrad@train/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2train/gradients/zlayer1/Wx_plus_b/add_grad/ReshapeReshape.train/gradients/zlayer1/Wx_plus_b/add_grad/Sum0train/gradients/zlayer1/Wx_plus_b/add_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
0train/gradients/zlayer1/Wx_plus_b/add_grad/Sum_1Sum*train/gradients/zlayer1/Relu_grad/ReluGradBtrain/gradients/zlayer1/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1Reshape0train/gradients/zlayer1/Wx_plus_b/add_grad/Sum_12train/gradients/zlayer1/Wx_plus_b/add_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
;train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_depsNoOp3^train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape5^train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1
�
Ctrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependencyIdentity2train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape<^train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape*'
_output_shapes
:���������

�
Etrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency_1Identity4train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1<^train/gradients/zlayer1/Wx_plus_b/add_grad/tuple/group_deps*
_output_shapes

:
*
T0*G
_class=
;9loc:@train/gradients/zlayer1/Wx_plus_b/add_grad/Reshape_1
�
4train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMulMatMulCtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependencyzlayer1/weights/Variable/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
6train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1MatMulinputs/x_inputCtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
>train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp5^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul7^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1
�
Ftrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity4train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul?^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Htrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity6train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1?^train/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/zlayer1/Wx_plus_b/MatMul_grad/MatMul_1*
_output_shapes

:

h
#train/GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
Jtrain/GradientDescent/update_zlayer1/weights/Variable/ApplyGradientDescentApplyGradientDescentzlayer1/weights/Variable#train/GradientDescent/learning_rateHtrain/gradients/zlayer1/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@zlayer1/weights/Variable*
_output_shapes

:

�
Itrain/GradientDescent/update_zlayer1/biases/Variable/ApplyGradientDescentApplyGradientDescentzlayer1/biases/Variable#train/GradientDescent/learning_rateEtrain/gradients/zlayer1/Wx_plus_b/add_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@zlayer1/biases/Variable*
_output_shapes

:

�
Jtrain/GradientDescent/update_zlayer2/weights/Variable/ApplyGradientDescentApplyGradientDescentzlayer2/weights/Variable#train/GradientDescent/learning_rateHtrain/gradients/zlayer2/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*+
_class!
loc:@zlayer2/weights/Variable
�
Itrain/GradientDescent/update_zlayer2/biases/Variable/ApplyGradientDescentApplyGradientDescentzlayer2/biases/Variable#train/GradientDescent/learning_rateEtrain/gradients/zlayer2/Wx_plus_b/add_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@zlayer2/biases/Variable
�
train/GradientDescentNoOpJ^train/GradientDescent/update_zlayer1/biases/Variable/ApplyGradientDescentK^train/GradientDescent/update_zlayer1/weights/Variable/ApplyGradientDescentJ^train/GradientDescent/update_zlayer2/biases/Variable/ApplyGradientDescentK^train/GradientDescent/update_zlayer2/weights/Variable/ApplyGradientDescent
�
initNoOp^zlayer1/biases/Variable/Assign ^zlayer1/weights/Variable/Assign^zlayer2/biases/Variable/Assign ^zlayer2/weights/Variable/Assign""%
train_op

train/GradientDescent"�
	variables��
�
zlayer1/weights/Variable:0zlayer1/weights/Variable/Assignzlayer1/weights/Variable/read:02zlayer1/weights/random_normal:08
s
zlayer1/biases/Variable:0zlayer1/biases/Variable/Assignzlayer1/biases/Variable/read:02zlayer1/biases/add:08
�
zlayer2/weights/Variable:0zlayer2/weights/Variable/Assignzlayer2/weights/Variable/read:02zlayer2/weights/random_normal:08
s
zlayer2/biases/Variable:0zlayer2/biases/Variable/Assignzlayer2/biases/Variable/read:02zlayer2/biases/add:08"�
	summaries�
�
zlayer1/weights/Weights1:0
zlayer1/biases/biases1:0
zlayer1/outputs1:0
zlayer2/weights/Weights2:0
zlayer2/biases/biases2:0
zlayer2/outputs2:0
loss/loss:0"�
trainable_variables��
�
zlayer1/weights/Variable:0zlayer1/weights/Variable/Assignzlayer1/weights/Variable/read:02zlayer1/weights/random_normal:08
s
zlayer1/biases/Variable:0zlayer1/biases/Variable/Assignzlayer1/biases/Variable/read:02zlayer1/biases/add:08
�
zlayer2/weights/Variable:0zlayer2/weights/Variable/Assignzlayer2/weights/Variable/read:02zlayer2/weights/random_normal:08
s
zlayer2/biases/Variable:0zlayer2/biases/Variable/Assignzlayer2/biases/Variable/read:02zlayer2/biases/add:08x1��      �lnB	|�w�6B�A*�
�
zlayer1/weights/Weights1*�	   @�&��    H�?      $@!  �[�j�)d�n��@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿӖ8��s��!������I�������g�骿�/�*>�?�g���w�?uo�p�?2g�G�A�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	    ����   @��?      $@!   ���?)+��w�?2�8/�C�ַ�� l(������g�骿�g���w���/�*>��`��a�8���uS��a���/����N�W�m?;8�clp?���g��?I���?����iH�?��]$A�?�{ �ǳ�?!�����?Ӗ8��s�?�Ca�G��?��7��?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?        
�
zlayer1/outputs1*�   �� @     p�@! �����@)�Qn_�y@2�        �-���q=ji6�9�?�S�F !?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �@              �?              �?               @              �?              �?              �?       @              @              �?      �?              �?      @      �?      �?      @      @      �?       @      @       @      @      @      &@      "@      (@      *@      ,@      ,@      1@      2@      4@      7@      8@      9@      ?@      ?@     �B@      C@      F@     @Q@     �R@      R@     �L@     �M@      O@      M@      P@     @Q@      F@      ?@     �A@     �C@      D@      F@      J@      E@     �F@      I@     �F@      D@      B@      D@      F@     �D@     �B@      >@      4@      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   @���   ��e�?      $@!   �r�@)���8�d@2�S�Fi���yL������+Se*8�\l�9⿗�7�ֿ�Ca�G�ԿI���?����iH�?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	    ��2�    ��2�      �?!    ��2�)  @r�~u>2�u�w74���82��������:              �?        
�

zlayer2/outputs2*�
	   ����    �D�?     �r@!  @0�FT�)�ݱ��J@2�2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab���"�uԖ�^�S����#�+(�ŉ�eiS�m��P}���h�Tw��Nof�o��5sz?���T}?�#�h/�?���&�?}Y�4j�?��<�A��?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�              &@      3@      1@      .@      ,@      <@      9@      8@      5@      3@      2@       @       @      @      �?       @       @      �?       @      �?      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?      �?      �?       @      �?       @       @       @       @      @      @      @      @      @      @      @      @      @      �?        

	loss/loss���>���a      q7S	���6B�A*�"
�
zlayer1/weights/Weights1*�	    �+��   ��j�?      $@!  �'���)�`uބ�@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%�W�i�bۿ�^��h�ؿ�?>8s2ÿӖ8��s������iH��I�������g��?I���?uo�p�?2g�G�A�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   `\&��    D%�?      $@!   �6��?)��`�]��?2��g���w���/�*>���7c_XY��#�+(�ŉ�o��5sz�*QH�x�E��{��^��m9�H�[��"�uԖ?}Y�4j�?�/��?�uS��a�?����iH�?��]$A�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�K?�?�Z�_���?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?        
�

zlayer1/outputs1*�
   @ @     p�@! P	⼴�@)+K�swx@2�        �-���q=1��a˲?6�]��?��d�r?�5�i}1?k�1^�sO?nK���LQ?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             t�@              �?              �?              �?              �?      �?      �?              �?       @              �?      �?      �?      �?       @      �?      @      @       @      �?       @      @      @      @      @      @      @      @      @      @      @      "@      0@      .@      0@      5@      5@      8@     �D@     �F@     �H@     �K@     �M@      P@     @R@     �S@     @V@     �W@     �Z@     �U@     �@@      =@      >@      A@      B@     �D@      G@      I@     �K@     �H@      I@     �L@     �I@      E@     �D@     �B@      A@     �A@      5@      "@      "@      "@      $@      (@      (@      ,@      @        
�
zlayer2/weights/Weights2*�	    �	��   @E��?      $@!   h�	@)��	��@2�S�Fi���yL������+Se*8�\l�9��Ca�G�Կ_&A�o�ҿ8/�C�ַ?%g�cE9�?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	    �ξ?    �ξ?      �?!    �ξ?)  �#���?2��(!�ؼ?!�����?�������:              �?        
�
zlayer2/outputs2*�	   ���Ϳ   ����?     �r@!  ����O@) ~`�OT@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� ��o��5sz�*QH�x�&b՞
�u�;8�clp��N�W�m���bB�SY�ܗ�SsW�a�Ϭ(�>8K�ߝ�>k�1^�sO?nK���LQ?E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?&b՞
�u?*QH�x?>	� �?����=��?#�+(�ŉ?�7c_XY�?�"�uԖ?}Y�4j�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�������:�              ,@      0@      (@      (@      &@      $@       @      "@      @      @      @      @      @      @      @      @      @      @      @       @      �?      @       @      �?      �?       @      �?      �?       @              �?               @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?      �?       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      "@      "@      $@      &@       @        

	loss/loss�M>��.A      k��	�d7B�A*�$
�
zlayer1/weights/Weights1*�	   @�$��    <��?      $@!   v��)���'��@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%�W�i�bۿ�^��h�ؿӖ8��s��!������� l(����{ �ǳ����]$A�?�{ �ǳ�?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   @����   ��I�?      $@!   n��?)��|��?2��{ �ǳ����]$A鱿�uS��a���/���}Y�4j���"�uԖ�^�S�����Rc�ݒ�ߤ�(g%k�P}���h��"�uԖ?}Y�4j�?I���?����iH�?� l(��?8/�C�ַ?��(!�ؼ?!�����?�Z�_���?����?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?        
�
zlayer1/outputs1*�   �� @     p�@! ��fB3�@)�l����w@2�        �-���q=�u�w74?��%�V6?��%>��:?d�\D�X=?�!�A?�T���C?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             �@              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?              @               @              @       @      �?      @      @      @      @      @      @      @      @       @       @      @      &@      $@      (@      &@      ,@      0@      ,@      8@      <@      @@      A@     �C@      F@      G@     �I@      K@     �N@     �Q@     �R@      T@      W@     @S@     �N@      K@      ?@     �A@      C@      E@      F@     �I@     �H@      G@     �I@      J@     �E@      C@     �C@      >@      @@     �B@      :@      @      "@      "@      &@      &@      (@      ,@      @        
�
zlayer2/weights/Weights2*�	    _G��   �i�?      $@!   ���@)�w��@2�S�Fi���yL������\l�9⿰1%��Ca�G�Կ_&A�o�ҿ����iH�?��]$A�?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   @�x�?   @�x�?      �?!   @�x�?) ��+S�?2Ӗ8��s�?�?>8s2�?�������:              �?        
�
zlayer2/outputs2*�	   �5�ҿ   ����?     �r@!  �&��C@)vGn�4	E@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� �����T}�o��5sz�*QH�x�uWy��r�;8�clp��l�P�`�E��{��^��m9�H�[�+A�F�&?I�I�)�(?���%��b?5Ucv0ed?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�������:�              @      @      @      @      @      @      @      @      "@      "@       @       @       @      @      @      @      @      @      @      @      @       @       @      @       @       @      @      �?      �?       @      �?      �?       @              �?      �?               @              �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?      �?      �?      �?      �?      �?       @       @      �?       @      @       @       @      @      @      @      @      @      �?              �?      �?      �?      �?      �?       @      �?       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @       @       @      $@      @        

	loss/lossq�>���$1      V��	�(1J7B�A*�$
�
zlayer1/weights/Weights1*�	   `$��    Y�?      $@!   ����)]�͜��@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿӖ8��s��!������8/�C�ַ�� l(���� l(��?8/�C�ַ?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	    2�    �h�?      $@!   @�m�?)�#���ɸ?2��{ �ǳ����]$A鱿�uS��a���/�����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ�}Y�4j�?��<�A��?I���?����iH�?��]$A�?�{ �ǳ�?Ӗ8��s�?�?>8s2�?�Z�_���?����?�������:�              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?        
�
zlayer1/outputs1*�   �g @     p�@! ��"��@).��?�w@2�        �-���q=��%>��:?d�\D�X=?���#@?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             |�@              �?      �?              �?              �?      �?              �?               @              �?      �?      �?       @       @       @      @      @      @       @       @      @       @      @      @      @      @      @      @      @      "@      @      $@      $@      &@      &@      ,@      .@      .@      2@      7@      ;@      A@     �A@     �C@      E@      H@      J@     �L@     �O@      Q@     @S@     �T@     @Q@      J@     �N@     �O@     �G@      C@      E@      G@      I@     �H@      G@      I@     �J@     �E@      B@      ?@      =@     �@@      B@      :@      @      "@      "@      &@      &@      (@      ,@      @        
�
zlayer2/weights/Weights2*�	   `�L��    F\�?      $@!   �ڢ@)�����@2�S�Fi���yL������\l�9⿰1%��Ca�G�Կ_&A�o�ҿI���?����iH�?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   ���?   ���?      �?!   ���?) ����c�?2�QK|:�?�@�"��?�������:              �?        
�
zlayer2/outputs2*�	   �WYͿ   �R��?     �r@!  P�lM@)�5nF@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�k�1^�sO�IcD���L���d�r?�5�i}1?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�������:�              @      @      @      @      @      @      @      @      @       @      @       @       @       @      �?      @      @      @       @      @       @       @      @      �?      �?       @      �?       @       @              �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @      �?      �?              @       @      �?       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      "@      "@      "@      @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @       @      "@      $@      @        

	loss/loss�w�=��@�      t��	��T7B�A*�!
�
zlayer1/weights/Weights1*�	   ��"��   �?��?      $@!   ����)��`GSm@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿ!��������(!�ؼ�8/�C�ַ�� l(���%g�cE9�?��(!�ؼ?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	    �o��    ��?      $@!   ���?)8��OT��?2�� l(����{ �ǳ���/�*>��`��a�8���uS��a���v��ab����<�A���}Y�4j���"�uԖ��"�uԖ?}Y�4j�?I���?����iH�?��]$A�?Ӗ8��s�?�?>8s2�?�Z�_���?����?�������:�              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?        
�
zlayer1/outputs1*�   `g @     p�@! �,C��@)�p5�ғw@2�        �-���q=��%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             Ȓ@               @              �?              �?               @              �?              �?      �?              @              �?      �?       @       @      �?       @       @      @      @      �?       @      @      @      @      @      @      @      @      @      @      "@       @      &@      "@      (@      ,@      ,@      ,@      3@      7@      <@      @@     �@@     �B@      E@      G@     �G@     �K@     �N@      Q@      R@     �R@     @P@     �H@      L@     �M@      Q@      K@     �E@      G@      I@      G@      G@     �I@      I@     �D@      B@      ;@      =@      @@      B@      :@      @      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   ��\��   @�I�?      $@!  ��]�@)+��h%�@2�S�Fi���yL�������1%���Z%�޿�Ca�G�Կ_&A�o�ҿ���g��?I���?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   @a�?   @a�?      �?!   @a�?)�������?2�@�"��?�K?�?�������:              �?        
�
zlayer2/outputs2*�	   ��h˿   ���?     �r@!  똯O@)Rv��D@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�������&���#�h/���7c_XY��-Ա�L�����J�\��*QH�x�&b՞
�u��N�W�m�ߤ�(g%k�5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?����=��?���J�\�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�������:�              @      @      @      @      @      @       @      @       @       @       @       @       @       @               @      �?       @              �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?      �?      �?      �?      �?      @      �?      �?      @      �?      @      @      @      @      @      @      @      @      @      @       @      @      "@       @      &@      &@      *@      ,@      $@       @      @      @      @      @      @      @      @      @      @      @      @      @       @      "@      "@      "@        

	loss/loss�E�=L�5z�      u�o0	���Z7B�A*�!
�
zlayer1/weights/Weights1*�	    D"��   ����?      $@!   xL��))�7Y@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿ!��������(!�ؼ�%g�cE9��8/�C�ַ���(!�ؼ?!�����?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   @�A��   `��?      $@!   ��<�?)�p�=<�?2�� l(����{ �ǳ��I�������g�骿�/�*>��`��a�8���v��ab����<�A���}Y�4j���"�uԖ?}Y�4j�?I���?����iH�?��]$A�?�?>8s2�?yD$��?�Z�_���?����?�������:�              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?        
�
zlayer1/outputs1*�   �� @     p�@! @����@)��0k�w@2�        �-���q=�.�?ji6�9�?�7Kaa+?��VlQ.?a�$��{E?
����G?�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             ��@              �?              �?              �?      �?      �?              �?      �?      �?              �?              �?       @              @               @       @       @       @      @      �?      @       @       @      @      @      @      @      @      @      @      "@      @      $@      $@      $@      $@      0@      ,@      .@      2@      7@      :@      @@      >@     �C@     �C@      F@      H@     �J@      N@     �O@     �Q@     @R@     �O@     �G@      J@     �L@     @P@     �Q@      K@      G@      J@      F@     �G@      J@     �G@      E@      <@      <@      =@      @@      B@      :@      @      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   �.f��   �J>�?      $@!   ���@)�;��@2�S�Fi���yL�������1%���Z%�޿�Ca�G�Կ_&A�o�ҿ�g���w�?���g��?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   ��2�?   ��2�?      �?!   ��2�?) �DO��?2�K?�?�Z�_���?�������:              �?        
�
zlayer2/outputs2*�	    _�ȿ    7��?     �r@!  `��sQ@)Ĩ}Qj�C@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ����&���#�h/���7c_XY��-Ա�L�����J�\��o��5sz�*QH�x�;8�clp��N�W�m����%��b?5Ucv0ed?Tw��Nof?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�������:�              @      @      @      @       @      @       @       @       @       @       @       @      �?      �?       @               @              �?      �?              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @               @       @      @      @      @       @      @      "@      "@      "@      (@      *@      0@      .@      1@      0@      @      @      @      @      @      @      @      @      @      @      @       @      "@      $@      @        

	loss/loss	�=�x5Á      u�a�	�{O\7B�A*� 
�
zlayer1/weights/Weights1*�	   ��!��   ��O�?      $@!  ����)G�'H@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿ!��������(!�ؼ�%g�cE9��8/�C�ַ���(!�ؼ?!�����?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   ��ﵿ   @�@�?      $@!   H���?)�C=ee��?2�8/�C�ַ�� l(�����]$A鱿����iH���/�*>��`��a�8���/����v��ab����<�A���}Y�4j���"�uԖ?}Y�4j�?I���?����iH�?��]$A�?yD$��?�QK|:�?�Z�_���?����?�������:�              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?        
�
zlayer1/outputs1*�   �� @     p�@!  �o��@)[rE=%|w@2�        �-���q=���#@?�!�A?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             �@              �?              �?              �?              �?              �?      �?      �?      �?       @      �?              �?       @              @      �?      @      �?      @      @      @      @      @      @      @      @      @      @      @      @       @       @      "@      "@      $@      ,@      ,@      ,@      ,@      4@      6@      7@     �@@      @@     �A@     �C@      F@      G@     �K@      M@      O@     @Q@      Q@      O@      G@     �I@      L@     �O@     �P@     �R@     �H@     �H@     �F@     �G@     �I@      G@      E@      8@      ;@      <@     �@@      B@      ;@      @      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   @-n��   ��4�?      $@!   ��@)�����@2�S�Fi���yL�������1%���Z%�޿�Ca�G�Կ_&A�o�ҿ�/�*>�?�g���w�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   `ja�?   `ja�?      �?!   `ja�?)@:��%Ű?2�Z�_���?����?�������:              �?        
�

zlayer2/outputs2*�
	   ��hǿ   �-�?     �r@!  �Ϸ�R@)}�L�C@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L��&b՞
�u�hyO�s�P}���h�Tw��Nof�ߤ�(g%k?�N�W�m?&b՞
�u?*QH�x?eiS�m�?#�+(�ŉ?�7c_XY�?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�������:�              �?      @      @      @      @      @       @       @      @      �?       @       @      �?      �?       @      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @               @       @      �?       @       @       @      @      @      "@      &@      (@      (@      ,@      0@      4@      3@      6@      "@      @      @      @      @      @      @      @      @       @       @      $@      $@      @        

	loss/loss�ك=އFMq      K��		|`7B�A*� 
�
zlayer1/weights/Weights1*�	    }!��   @N�?      $@!   �V��)�C�3�9@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿ!��������(!�ؼ�%g�cE9��8/�C�ַ�!�����?Ӗ8��s�?+Se*8�?uo�p�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   @�k��   @���?      $@!   b"��?)8�'~��?2�8/�C�ַ�� l(����{ �ǳ����]$A鱿�g���w���/�*>���/����v��ab����<�A����"�uԖ?}Y�4j�?I���?����iH�?��]$A�?yD$��?�QK|:�?�Z�_���?����?�������:�              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?        
�
zlayer1/outputs1*�    � @     p�@!  s`N�@)^���~w@2�        �-���q=I�I�)�(?�7Kaa+?��%>��:?d�\D�X=?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             �@              �?              �?              �?              �?      �?              �?      �?      �?      �?              �?      �?              @              @      �?      @       @       @      �?      @       @       @      @       @      @      @      @      @      @      @      "@       @      "@      "@      &@      $@      0@      *@      1@      2@      5@      ;@      <@      @@     �B@      D@     �D@     �F@      K@      L@      O@     �Q@      P@     �N@      G@     �H@     �K@     �N@     �P@     @R@     �P@      I@      E@     �G@      K@     �F@     �B@      8@      ;@      <@     �@@     �A@      <@      @      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   @,t��   �.�?      $@!  ��|�@)VkC�@2�S�Fi���yL�������1%���Z%�޿�Ca�G�Կ_&A�o�ҿ`��a�8�?�/�*>�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	    �w�?    �w�?      �?!    �w�?)@�GS��?2����?_&A�o��?�������:              �?        
�

zlayer2/outputs2*�
	    �+ƿ   ����?     �r@!  @���S@)�j8��D@2��QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ����&���#�h/������=���>	� �����T}�U�4@@�$?+A�F�&?k�1^�sO?nK���LQ?>	� �?����=��?���J�\�?-Ա�L�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�������:�              @      @      @      @       @      @       @       @       @       @       @      �?      �?       @               @              �?      �?              �?      �?               @              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              �?      �?      �?       @      �?       @       @       @       @      @      @      @      @      @      &@      .@      0@      1@      6@      8@      9@      &@      @      @      @      @      @      @       @      @      "@      "@      &@      �?        

	loss/loss^�k=����1      J��T	�cd7B�A*� 
�
zlayer1/weights/Weights1*�	   �E!��    ���?      $@!  �N��)���-@2�ܔ�.�u��S�Fi���cI���+�;$�2g�G�A�uo�p�\l�9⿰1%��^��h�ؿ��7�ֿ!��������(!�ؼ�%g�cE9��8/�C�ַ�Ӗ8��s�?�?>8s2�?\l�9�?+Se*8�?�������:�              �?              �?              �?               @              �?              �?              �?              �?              �?        
�
zlayer1/biases/biases1*�	   @Ķ�   `Q��?      $@!   �x��?)�z�|S��?2�8/�C�ַ�� l(����{ �ǳ���g���w���/�*>���/����v��ab����<�A����"�uԖ?}Y�4j�?I���?����iH�?��]$A�?�QK|:�?�@�"��?�Z�_���?����?�������:�              �?      �?              �?              �?      �?              �?              �?      �?              �?              �?        
�
zlayer1/outputs1*�   @o @     p�@! �À��@)��`��w@2�        �-���q=I��P=�>��Zr[v�>I�I�)�(?�7Kaa+?���#@?�!�A?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�             $�@              �?              �?              �?              �?              �?              �?               @       @              �?       @               @      �?      @      �?      �?      @      @      �?       @      @       @      @      @      @      @      @      @      @       @       @      @      &@      (@      "@      .@      .@      .@      3@      6@      :@      >@      =@      B@     �C@      E@      G@     �I@     �L@     �O@     �P@     �O@      O@      F@     �H@     �J@      N@      P@      R@     �S@      J@      F@     �G@      J@     �F@     �A@      7@      :@      =@      @@     �B@      <@      @      "@      "@      &@      &@      *@      *@      @        
�
zlayer2/weights/Weights2*�	   `�x��   ��(�?      $@!  �H��@)@�.�<�@2�S�Fi���yL�������1%���Z%�޿�Ca�G�Կ_&A�o�ҿ�uS��a�?`��a�8�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�������:�              �?              �?              �?              �?              �?              �?              �?       @              �?        
{
zlayer2/biases/biases2*a	   `,c�?   `,c�?      �?!   `,c�?)@���!�?2����?_&A�o��?�������:              �?        
�

zlayer2/outputs2*�
	   ��:ſ   �~�?     �r@!  ��\T@)��}��lD@2��QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����#�h/���7c_XY��o��5sz�*QH�x�E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�