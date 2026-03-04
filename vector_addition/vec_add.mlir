func.func @parallel_vector_add(
  %arg0: memref<1024xf32>, 
  %arg1: memref<1024xf32>, 
  %arg2: memref<1024xf32>) {
  // This is the "High Level" parallel intent
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, // Access arg0[d0]
      affine_map<(d0) -> (d0)>, // Access arg1[d0]
      affine_map<(d0) -> (d0)>  // Access arg2[d0]
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>)
    outs(%arg2 : memref<1024xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %0 = arith.addf %in1, %in2 : f32
      linalg.yield %0 : f32
  }
  return
}
